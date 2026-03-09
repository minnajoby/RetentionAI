from flask import Flask, render_template, request, send_file
from datetime import datetime
import joblib
import numpy as np
import sqlite3
import os
import io
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.platypus import KeepTogether

plt.switch_backend('Agg')
app = Flask(__name__)

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
os.environ["HF_TOKEN"] = "your_huggingface_token_here"

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  time_stamp TEXT, model_used TEXT, probability TEXT, result TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- GLOBAL MODELS & SCALER ---
models = {}
scaler_7 = None

def load_models():
    global scaler_7
    try:
        scaler_7 = joblib.load('models/scaler_7.pkl')
        models["CatBoost"] = joblib.load('models/catboost_7.pkl')
        models["LightGBM"] = joblib.load('models/lightgbm_7.pkl')
        models["TabPFN"] = joblib.load('models/tabpfn_7.pkl')
        tn = TabNetClassifier()
        tn.load_model('models/tabnet_7.zip')
        models["TabNet"] = tn
        print("--- ALL 7-FEATURE MODELS AND SCALER READY ---")
    except Exception as e:
        print(f"--- MODEL LOADING ERROR: {e} ---")

load_models()

# --- HELPER: Geography/Gender display ---
GEO_MAP   = {"0": "France", "1": "Germany", "2": "Spain"}
GEN_MAP   = {"1": "Male", "0": "Female"}

FEATURE_NAMES  = ['Geography','Gender','Age','Tenure','Balance','NumOfProducts','IsActiveMember']
DISPLAY_NAMES  = {
    'Geography': 'Location', 'Gender': 'Gender', 'Age': 'Customer Age',
    'Tenure': 'Tenure', 'Balance': 'Account Balance',
    'NumOfProducts': 'Product Count', 'IsActiveMember': 'Membership Activity'
}

# ── RETENTION OFFER RULES ──────────────────────────────────
# Each rule maps a top-risk SHAP feature to a personalized offer card.
# icon: FontAwesome class, urgency: critical/high/medium, impact: estimated churn reduction
OFFER_RULES = {
    'NumOfProducts': {
        'title':       'Cross-Sell Product Bundle',
        'description': 'Customer holds fewer products than average. Offer a bundled savings + investment account with fee waiver for 6 months. Multi-product holders churn 3× less.',
        'action':      'Assign relationship manager for product review call within 48 hrs',
        'icon':        'fa-layer-group',
        'urgency':     'critical',
        'impact':      '↓ 38% churn risk',
        'color':       '#dc2626',
    },
    'IsActiveMember': {
        'title':       'Re-Engagement Campaign',
        'description': 'Inactive members are at 3× higher churn risk. Trigger a personalized re-engagement email with cashback on next 3 transactions and a loyalty points bonus.',
        'action':      'Enroll in 90-day re-engagement drip campaign immediately',
        'icon':        'fa-fire',
        'urgency':     'critical',
        'impact':      '↓ 31% churn risk',
        'color':       '#dc2626',
    },
    'Age': {
        'title':       'Senior Loyalty Programme',
        'description': 'Customers aged 45–60 show the highest churn propensity. Offer a premium account tier with priority support, zero fees, and a dedicated advisor.',
        'action':      'Upgrade to Premier tier and schedule welcome advisor call',
        'icon':        'fa-crown',
        'urgency':     'high',
        'impact':      '↓ 24% churn risk',
        'color':       '#d97706',
    },
    'Balance': {
        'title':       'High-Yield Savings Offer',
        'description': 'Account balance is a significant risk driver. Offer a competitive fixed-deposit rate (0.5% above market) to lock in funds and deepen financial commitment.',
        'action':      'Send personalised rate offer via app notification within 24 hrs',
        'icon':        'fa-piggy-bank',
        'urgency':     'high',
        'impact':      '↓ 19% churn risk',
        'color':       '#d97706',
    },
    'Tenure': {
        'title':       'Early Tenure Retention Bonus',
        'description': 'Shorter-tenure customers are significantly more likely to churn. Offer a milestone reward at the 1-year mark — bonus interest or cashback credit.',
        'action':      'Schedule automated milestone reward at 12-month anniversary',
        'icon':        'fa-calendar-check',
        'urgency':     'medium',
        'impact':      '↓ 15% churn risk',
        'color':       '#0891b2',
    },
    'Gender': {
        'title':       'Personalised Banking Experience',
        'description': 'Demographic signals indicate this segment benefits from tailored communication. Adjust product messaging and support channels to match user preferences.',
        'action':      'Update communication profile and assign segment-specific advisor',
        'icon':        'fa-user-check',
        'urgency':     'medium',
        'impact':      '↓ 11% churn risk',
        'color':       '#0891b2',
    },
    'Geography': {
        'title':       'Regional Retention Offer',
        'description': 'This customer\'s region shows above-average churn rates. Offer region-specific perks: local branch priority access, regional cashback partnerships, or language support.',
        'action':      'Enrol in regional loyalty programme and notify via preferred channel',
        'icon':        'fa-map-location-dot',
        'urgency':     'medium',
        'impact':      '↓ 10% churn risk',
        'color':       '#0891b2',
    },
}

URGENCY_ORDER = {'critical': 0, 'high': 1, 'medium': 2}

def get_retention_offers(sorted_impacts, top_n=3):
    """Return up to top_n offer dicts ranked by SHAP magnitude, risk factors only."""
    offers = []
    for feat, val in sorted_impacts:
        if val > 0 and feat in OFFER_RULES:   # only features pushing risk UP
            offers.append(OFFER_RULES[feat])
        if len(offers) >= top_n:
            break
    # If fewer than 2 risk offers, pad with the highest-magnitude stable features
    if len(offers) < 2:
        for feat, val in sorted_impacts:
            if val <= 0 and feat in OFFER_RULES and OFFER_RULES[feat] not in offers:
                offers.append(OFFER_RULES[feat])
            if len(offers) >= top_n:
                break
    return offers


def get_history():
    conn = sqlite3.connect('history.db'); c = conn.cursor()
    c.execute("SELECT time_stamp, model_used, probability, result FROM predictions ORDER BY id DESC LIMIT 5")
    h = c.fetchall(); conn.close()
    return h

@app.route('/')
def home():
    return render_template('index.html', inputs=None, history=get_history())

@app.route('/predict', methods=['POST'])
def predict():
    history = get_history()
    try:
        choice = request.form.get('ModelChoice', 'CatBoost')
        current_model = models.get(choice, models['CatBoost'])

        # 1. Validation
        feature_list = []
        for f in FEATURE_NAMES:
            if f == 'IsActiveMember':
                val = request.form.get(f, '0')  # unchecked checkbox = 0
            else:
                val = request.form.get(f)
                if val is None or val.strip() == "":
                    return render_template('index.html', error_msg="All fields are required.",
                                           history=history, inputs=request.form)
            feature_list.append(float(val))

        # 2. Process
        raw_features = np.array(feature_list).reshape(1, -1)
        scaled_features = scaler_7.transform(raw_features)
        if choice in ["TabNet", "TabPFN"]:
            scaled_features = scaled_features.astype(np.float32)

        # 3. Predict
        prediction  = current_model.predict(scaled_features)
        prob_value  = float(current_model.predict_proba(scaled_features)[0][1])
        res_text    = "HIGH RISK" if prediction[0] == 1 else "LOW RISK"
        prob_display = f"{prob_value * 100:.2f}%"

        # 4. SHAP waterfall (always CatBoost explainer for consistency)
        explainer = shap.TreeExplainer(models["CatBoost"])
        X_explain = pd.DataFrame(scaled_features, columns=FEATURE_NAMES)
        shap_obj  = explainer(X_explain)

        plt.figure(figsize=(8, 4))
        shap.plots.waterfall(shap_obj[0], show=False)
        plt.title("Risk Factor Decomposition", pad=20, fontsize=10, fontweight='bold')
        plt.savefig('static/local_explanation.png', bbox_inches='tight', dpi=100)
        plt.close()

        # 5. SHAP values for breakdown table
        shap_vals   = shap_obj.values[0]
        impact_dict = dict(zip(FEATURE_NAMES, shap_vals))
        sorted_impacts = sorted(impact_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        top_risk    = max(impact_dict, key=impact_dict.get)
        top_stable  = min(impact_dict, key=impact_dict.get)
        risk_reason   = f"The primary driver for churn risk is {DISPLAY_NAMES[top_risk]}."
        stable_reason = f"The strongest factor for retention is {DISPLAY_NAMES[top_stable]}."

        # 6. Build human-readable input summary for template & PDF
        raw = request.form
        input_summary = {
            'Model':       choice,
            'Geography':   GEO_MAP.get(raw.get('Geography',''), raw.get('Geography','')),
            'Gender':      GEN_MAP.get(raw.get('Gender',''),    raw.get('Gender','')),
            'Age':         raw.get('Age',''),
            'Balance':     f"${float(raw.get('Balance',0)):,.2f}",
            'Tenure':      f"{raw.get('Tenure','')} years",
            'Products':    raw.get('NumOfProducts',''),
            'Active':      'Yes' if raw.get('IsActiveMember') else 'No',
        }

        # 7. DB log
        current_time = datetime.now().strftime("%H:%M:%S")
        conn = sqlite3.connect('history.db'); c = conn.cursor()
        c.execute("INSERT INTO predictions (time_stamp, model_used, probability, result) VALUES (?,?,?,?)",
                  (current_time, choice, prob_display, res_text))
        conn.commit()
        history = get_history(); conn.close()

        # 8. Generate retention offers
        retention_offers = get_retention_offers(sorted_impacts)

        return render_template('index.html',
                               prediction_text  = res_text,
                               prob_text        = prob_display,
                               inputs           = request.form,
                               history          = history,
                               risk_reason      = risk_reason,
                               stable_reason    = stable_reason,
                               sorted_impacts   = sorted_impacts,
                               display_names    = DISPLAY_NAMES,
                               input_summary    = input_summary,
                               timestamp        = current_time,
                               retention_offers = retention_offers)

    except Exception as e:
        return render_template('index.html', error_msg=f"System Error: {str(e)}", history=history)


# ─────────────────────────────────────────────────────────
#  PDF REPORT ROUTE
# ─────────────────────────────────────────────────────────
@app.route('/report', methods=['POST'])
def generate_report():
    """Generate and auto-download a styled PDF risk report."""
    try:
        pred_text     = request.form.get('prediction_text', '')
        prob_text     = request.form.get('prob_text', '')
        risk_reason   = request.form.get('risk_reason', '')
        stable_reason = request.form.get('stable_reason', '')
        timestamp     = request.form.get('timestamp', datetime.now().strftime("%H:%M:%S"))
        model_used    = request.form.get('model_used', '')
        date_str      = datetime.now().strftime("%B %d, %Y")

        geography = GEO_MAP.get(request.form.get('Geography',''), request.form.get('Geography',''))
        gender    = GEN_MAP.get(request.form.get('Gender',''),    request.form.get('Gender',''))
        age       = request.form.get('Age','')
        balance   = f"${float(request.form.get('Balance', 0)):,.2f}"
        tenure    = request.form.get('Tenure','')
        products  = request.form.get('NumOfProducts','')
        active    = 'Yes' if request.form.get('IsActiveMember') else 'No'

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=16*mm, rightMargin=16*mm,
                                topMargin=14*mm, bottomMargin=14*mm)

        NAVY    = colors.HexColor('#003366')
        NAVY2   = colors.HexColor('#004080')
        GOLD    = colors.HexColor('#c9a84c')
        DANGER  = colors.HexColor('#dc2626')
        SUCCESS = colors.HexColor('#059669')
        LIGHT   = colors.HexColor('#f4f6fb')
        BORDER  = colors.HexColor('#e4e9f2')
        MUTED   = colors.HexColor('#7a8fa6')
        WHITE   = colors.white
        is_high = pred_text == "HIGH RISK"
        VERDICT_COLOR = DANGER if is_high else SUCCESS

        W = A4[0] - 32*mm

        def S(name, **kw):
            return ParagraphStyle(name, **kw)

        title_s   = S('T',  fontName='Helvetica-Bold',  fontSize=15, textColor=WHITE,    leading=19)
        brand_s   = S('BR', fontName='Helvetica-Bold',  fontSize=10, textColor=GOLD,     leading=13)
        sub_s     = S('SB', fontName='Helvetica',       fontSize=7.5,textColor=colors.HexColor('#b0c4d8'), leading=11)
        section_s = S('H',  fontName='Helvetica-Bold',  fontSize=7.5,textColor=NAVY,     leading=11, spaceAfter=5, spaceBefore=12, textTransform='uppercase', letterSpacing=1.0)
        label_s   = S('L',  fontName='Helvetica-Bold',  fontSize=7,  textColor=MUTED,    leading=9,  textTransform='uppercase', letterSpacing=0.7)
        val_s     = S('V',  fontName='Helvetica-Bold',  fontSize=10, textColor=colors.HexColor('#0d1b2a'), leading=13)
        insight_s = S('I',  fontName='Helvetica',       fontSize=9,  textColor=colors.HexColor('#0d1b2a'), leading=13)
        prob_s    = S('P',  fontName='Helvetica-Bold',  fontSize=26, textColor=VERDICT_COLOR, leading=30, alignment=2)
        prob_lbl  = S('PL', fontName='Helvetica',       fontSize=7.5,textColor=colors.HexColor('#b0c4d8'), leading=10, alignment=2)
        pill_s    = S('PI', fontName='Helvetica-Bold',  fontSize=8,  textColor=WHITE,    leading=10, alignment=1)
        footer_s  = S('F',  fontName='Helvetica',       fontSize=7,  textColor=MUTED,    alignment=1, leading=10)

        story = []

        # ── HEADER ──
        pill_color = DANGER if is_high else SUCCESS
        pill_label = 'HIGH RISK' if is_high else 'LOW RISK'

        pill_cell = Table(
            [[Paragraph(pill_label, pill_s)]],
            colWidths=[60],
            style=TableStyle([
                ('BACKGROUND',(0,0),(0,0), pill_color),
                ('ROUNDEDCORNERS',[3,3,3,3]),
                ('TOPPADDING',(0,0),(0,0),5),('BOTTOMPADDING',(0,0),(0,0),5),
                ('LEFTPADDING',(0,0),(0,0),8),('RIGHTPADDING',(0,0),(0,0),8),
                ('ALIGN',(0,0),(0,0),'RIGHT'),
            ]))

        left_header = Table(
            [[Paragraph('RetentionAI', brand_s)],
             [Paragraph('Enterprise Churn Risk Report', title_s)],
             [Spacer(1, 4)],
             [Paragraph(f'Generated {date_str} at {timestamp}  |  Model: {model_used}', sub_s)]],
            colWidths=[W * 0.60],
            style=TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                              ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),
                              ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))

        right_header = Table(
            [[Paragraph(prob_text, prob_s)],
             [Paragraph('Churn Probability', prob_lbl)],
             [Spacer(1, 5)],
             [pill_cell]],
            colWidths=[W * 0.40],
            style=TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                              ('ALIGN',(0,0),(-1,-1),'RIGHT'),
                              ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),
                              ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))

        header_table = Table(
            [[left_header, right_header]],
            colWidths=[W * 0.60, W * 0.40],
            style=TableStyle([
                ('BACKGROUND',(0,0),(-1,-1),NAVY),
                ('TOPPADDING',(0,0),(-1,-1),16),('BOTTOMPADDING',(0,0),(-1,-1),16),
                ('LEFTPADDING',(0,0),(0,-1),18),('RIGHTPADDING',(-1,0),(-1,-1),18),
                ('ROUNDEDCORNERS',[8,8,8,8]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ]))
        story.append(header_table)
        story.append(Spacer(1, 10))

        # ── CUSTOMER PROFILE ──
        story.append(Paragraph('Customer Profile', section_s))

        def field_cell(lbl, v):
            return Table([[Paragraph(lbl, label_s)],[Paragraph(str(v), val_s)]],
                         style=TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),
                                           ('BOX',(0,0),(-1,-1),0.5,BORDER),
                                           ('ROUNDEDCORNERS',[5,5,5,5]),
                                           ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
                                           ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))

        cw4 = (W - 9) / 4
        story.append(Table([[field_cell('Geography', geography), field_cell('Gender', gender),
                              field_cell('Age', age),            field_cell('Active Member', active)]],
                           colWidths=[cw4]*4,
                           style=TableStyle([('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)])))
        story.append(Spacer(1, 5))
        cw3 = (W - 6) / 3
        story.append(Table([[field_cell('Account Balance', balance),
                              field_cell('Tenure', f'{tenure} years'),
                              field_cell('Products Held', products)]],
                           colWidths=[cw3]*3,
                           style=TableStyle([('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)])))
        story.append(Spacer(1, 4))

        # ── AI INSIGHTS ──
        story.append(Paragraph('Strategic AI Summary', section_s))

        def insight_cell(lbl, txt, bg, bdr, accent):
            il = ParagraphStyle('il2', fontName='Helvetica-Bold', fontSize=6.5, textColor=accent,
                                leading=9, textTransform='uppercase', letterSpacing=1.0)
            return Table([[Paragraph(lbl, il)],[Paragraph(txt, insight_s)]],
                         style=TableStyle([('BACKGROUND',(0,0),(-1,-1),bg),
                                           ('BOX',(0,0),(-1,-1),0.5,bdr),
                                           ('LINEBEFORE',(0,0),(0,-1),2.5,accent),
                                           ('ROUNDEDCORNERS',[4,4,4,4]),
                                           ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
                                           ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))

        iw = (W - 8) / 2
        story.append(Table(
            [[insight_cell('Primary Risk Factor', risk_reason,
                           colors.HexColor('#fef2f2'), colors.HexColor('#fecaca'), DANGER),
              insight_cell('Retention Strength',  stable_reason,
                           colors.HexColor('#f0fdf4'), colors.HexColor('#bbf7d0'), SUCCESS)]],
            colWidths=[iw, iw],
            style=TableStyle([('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)])))
        story.append(Spacer(1, 4))

        # ── SHAP CHART ──
        shap_path = 'static/local_explanation.png'
        if os.path.exists(shap_path):
            story.append(Paragraph('SHAP Risk Factor Decomposition', section_s))
            img_w = W
            img_h = img_w * 0.44
            img = RLImage(shap_path, width=img_w, height=img_h)
            story.append(Table([[img]], colWidths=[W],
                               style=TableStyle([
                                   ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#fafbfd')),
                                   ('BOX',(0,0),(-1,-1),0.5,BORDER),
                                   ('ROUNDEDCORNERS',[6,6,6,6]),
                                   ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                                   ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
                               ])))
            story.append(Spacer(1, 8))

        # ── RETENTION OFFERS ──
        offers_raw = request.form.get('offers_data', '')
        if offers_raw:
            story.append(Paragraph('Personalised Retention Offers', section_s))
            offer_list = [o.split('|||') for o in offers_raw.split('<<<') if '|||' in o]

            URGENCY_COLORS = {
                'critical': (colors.HexColor('#fef2f2'), colors.HexColor('#fecaca'), DANGER),
                'high':     (colors.HexColor('#fffbeb'), colors.HexColor('#fde68a'), colors.HexColor('#d97706')),
                'medium':   (colors.HexColor('#ecfeff'), colors.HexColor('#a5f3fc'), colors.HexColor('#0891b2')),
            }

            offer_title_s  = ParagraphStyle('OT', fontName='Helvetica-Bold', fontSize=9,  textColor=colors.HexColor('#0d1b2a'), leading=12)
            offer_badge_s  = ParagraphStyle('OB', fontName='Helvetica-Bold', fontSize=6.5, leading=9, textTransform='uppercase', letterSpacing=0.5)
            offer_desc_s   = ParagraphStyle('OD', fontName='Helvetica',      fontSize=8,  textColor=colors.HexColor('#3d5166'), leading=12)
            offer_action_s = ParagraphStyle('OA', fontName='Helvetica-Bold', fontSize=7.5, textColor=colors.HexColor('#0d1b2a'), leading=11)
            impact_s       = ParagraphStyle('OI', fontName='Helvetica-Bold', fontSize=6.5, textColor=SUCCESS, leading=9)

            ow = (W - (len(offer_list)-1)*6) / max(len(offer_list), 1)
            offer_cells = []
            for parts in offer_list:
                if len(parts) < 5:
                    continue
                otitle, ourgency, oimpact, odesc, oaction = parts[0], parts[1].strip(), parts[2], parts[3], parts[4]
                bg, bdr, accent = URGENCY_COLORS.get(ourgency, URGENCY_COLORS['medium'])
                cell = Table([
                    [Paragraph(otitle, offer_title_s)],
                    [Table([[Paragraph(ourgency.upper(), ParagraphStyle('ub', fontName='Helvetica-Bold', fontSize=6, textColor=accent, leading=8, textTransform='uppercase')),
                             Paragraph(oimpact, impact_s)]],
                           colWidths=[ow*0.38, ow*0.55],
                           style=TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),
                                             ('TOPPADDING',(0,0),(-1,-1),0),('BOTTOMPADDING',(0,0),(-1,-1),0),
                                             ('VALIGN',(0,0),(-1,-1),'MIDDLE')]))],
                    [Paragraph(odesc, offer_desc_s)],
                    [Table([[Paragraph(f'→  {oaction}', offer_action_s)]],
                           style=TableStyle([('BACKGROUND',(0,0),(0,0),bg),
                                             ('BOX',(0,0),(0,0),0.5,bdr),
                                             ('ROUNDEDCORNERS',[3,3,3,3]),
                                             ('LEFTPADDING',(0,0),(0,0),8),('RIGHTPADDING',(0,0),(0,0),8),
                                             ('TOPPADDING',(0,0),(0,0),5),('BOTTOMPADDING',(0,0),(0,0),5)]))],
                ], style=TableStyle([
                    ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#fafbfd')),
                    ('BOX',(0,0),(-1,-1),0.5,bdr),
                    ('LINEBEFORE',(0,0),(0,-1),2.5,accent),
                    ('ROUNDEDCORNERS',[4,4,4,4]),
                    ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
                    ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),
                ]))
                offer_cells.append(cell)

            if offer_cells:
                col_w = [(W - 6*(len(offer_cells)-1)) / len(offer_cells)] * len(offer_cells)
                gap_style = TableStyle([('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
                                        ('VALIGN',(0,0),(-1,-1),'TOP')])
                story.append(Table([offer_cells], colWidths=col_w, style=gap_style))
                story.append(Spacer(1, 8))

        # ── FOOTER ──
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceAfter=5))
        story.append(Paragraph(
            f'RetentionAI  |  Confidential Risk Assessment  |  {date_str}  |  Powered by CatBoost · LightGBM · TabNet · TabPFN',
            footer_s))

        doc.build(story)
        buf.seek(0)
        filename = f"RetentionAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name=filename)

    except Exception as e:
        return f"PDF generation error: {str(e)}", 500


if __name__ == "__main__":
    app.run(debug=True)