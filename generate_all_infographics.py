import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import numpy as np

# Set global styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica', 'sans-serif']

def create_chapter_mapping_infographic(filename):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('#F8F9FA')

    # Title
    plt.text(50, 95, "FYDP REPORT CHAPTER MAPPING GUIDE", ha='center', va='center', fontsize=24, fontweight='bold', color='#2C3E50')
    plt.text(50, 90, "How to map Problem Attributes (A), Program Outcomes (PO), and Knowledge Profiles (K) across chapters", 
             ha='center', va='center', fontsize=12, style='italic', color='#7F8C8D')

    chapters = [
        {"title": "Chapter 1: Intro", "desc": "Define the complex problem using first principles.", 
         "attr": "A2: Conflicting Req.\nA4: Familiarity", "po": "PO2: Problem Analysis", "k": "K1: Natural Sciences\nK2: Mathematics", "color": "#E3F2FD", "edge": "#1976D2"},
        {"title": "Chapter 2: Lit Review", "desc": "Deep math/science/theory & specialist research.", 
         "attr": "A1: Depth of Knowledge", "po": "PO1: Eng. Knowledge", "k": "K4: Specialist Knowledge\nK8: Research Literature", "color": "#E8F5E9", "edge": "#388E3C"},
        {"title": "Chapter 3: Methodology", "desc": "Design practice, codes, and project management.", 
         "attr": "A5: Applicable Codes\nA7: Interdependence", "po": "PO11: Project Mgt.", "k": "K3: Eng. Fundamentals\nK5: Design Practice", "color": "#FFF3E0", "edge": "#F57C00"},
        {"title": "Chapter 5: Complex Eng.", "desc": "Core analysis, simulation, and conclusions.", 
         "attr": "A3: Depth of Analysis\nA6: Stakeholders", "po": "PO4: Investigation\nPO12: Life-long Learning", "k": "K7: Sustainability / Impact", "color": "#FCE4EC", "edge": "#C2185B"}
    ]

    y_pos = 79
    for ch in chapters:
        # Draw Chapter Box
        rect = mpatches.FancyBboxPatch((5, y_pos-10), 22, 14, boxstyle="round,pad=1,rounding_size=2", facecolor=ch["color"], edgecolor=ch["edge"], lw=2.5)
        ax.add_patch(rect)
        ax.text(16, y_pos-1, ch["title"], ha='center', va='center', fontsize=16, fontweight='bold', color='#1A252F')
        ax.text(16, y_pos-6, textwrap.fill(ch["desc"], 25), ha='center', va='center', fontsize=10, color='#34495E')
        
        # Connectors
        ax.plot([28, 35], [y_pos-3, y_pos-3], color=ch["edge"], lw=2.5, zorder=1)
        ax.plot([55, 60], [y_pos-3, y_pos-3], color=ch["edge"], lw=2.5, zorder=1)
        ax.plot([80, 85], [y_pos-3, y_pos-3], color=ch["edge"], lw=2.5, zorder=1)
        
        # Attributes Box
        rect_a = mpatches.FancyBboxPatch((35, y_pos-9), 20, 12, boxstyle="round,pad=1,rounding_size=1", facecolor='#FFFFFF', edgecolor='#BDC3C7', lw=1.5)
        ax.add_patch(rect_a)
        ax.text(45, y_pos+1, "ATTRIBUTES (A)", ha='center', va='center', fontsize=10, fontweight='bold', color='#7F8C8D')
        ax.text(45, y_pos-4, ch["attr"], ha='center', va='center', fontsize=11, color='#2C3E50')

        # PO Box
        rect_po = mpatches.FancyBboxPatch((60, y_pos-9), 20, 12, boxstyle="round,pad=1,rounding_size=1", facecolor='#FFFFFF', edgecolor='#BDC3C7', lw=1.5)
        ax.add_patch(rect_po)
        ax.text(70, y_pos+1, "OUTCOMES (PO)", ha='center', va='center', fontsize=10, fontweight='bold', color='#7F8C8D')
        ax.text(70, y_pos-4, ch["po"], ha='center', va='center', fontsize=11, color='#2C3E50')

        # K Box
        rect_k = mpatches.FancyBboxPatch((85, y_pos-9), 13, 12, boxstyle="round,pad=1,rounding_size=1", facecolor='#FFFFFF', edgecolor='#BDC3C7', lw=1.5)
        ax.add_patch(rect_k)
        ax.text(91.5, y_pos+1, "KNOWLEDGE (K)", ha='center', va='center', fontsize=10, fontweight='bold', color='#7F8C8D')
        ax.text(91.5, y_pos-4, ch["k"], ha='center', va='center', fontsize=10, color='#2C3E50')
        
        y_pos -= 19

    # PO10 Footer
    rect_bottom = mpatches.FancyBboxPatch((5, 1), 93, 7, boxstyle="round,pad=1,rounding_size=2", facecolor='#E0F7FA', edgecolor='#0097A7', lw=2)
    ax.add_patch(rect_bottom)
    ax.text(51.5, 5.5, "PO10: COMMUNICATION", ha='center', va='center', fontsize=14, fontweight='bold', color='#006064')
    ax.text(51.5, 2.5, "Applies to the Entire Paper (Satisfied by the quality of your writing, formatting, and presentation)", ha='center', va='center', fontsize=11, color='#004D40')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

def create_combined_tables_infographic(filename):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#F8F9FA')
    
    # Left table 5.1
    cat_p = ['P1: Depth of Knowledge', 'P2: Conflicting Req.', 'P3: Depth of Analysis', 
             'P4: Familiarity of Issues', 'P5: Applicable Codes', 'P6: Stakeholder Involv.', 'P7: Interdependence']
    val_p = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    
    # Right table 5.2
    cat_a = ['A1: Range of Resources', 'A2: Level of Interaction', 'A3: Innovation', 
             'A4: Consequences for Society', 'A5: Familiarity']
    val_a = [1.0, 1.0, 1.0, 1.0, 0.5]

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1])
    
    # 5.1 Bar
    ax1 = fig.add_subplot(gs[0, 0])
    colors_p = ['#2ECC71' if v == 1.0 else '#F39C12' for v in val_p]
    bars1 = ax1.barh(cat_p, val_p, color=colors_p, height=0.6, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax1.set_xlim(0, 1.25)
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(['Not Addressed', 'Partial', 'Satisfied'], fontweight='bold', color='#7F8C8D')
    ax1.set_title("Table 5.1: Mapping with Complex Problem Solving", fontsize=16, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    for bar, val in zip(bars1, val_p):
        lbl = 'Satisfied' if val == 1.0 else 'Partial'
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, lbl, va='center', fontweight='bold', color='#2C3E50', fontsize=12)
        
    # 5.2 Bar
    ax2 = fig.add_subplot(gs[1, 0])
    colors_a = ['#2ECC71' if v == 1.0 else '#F39C12' for v in val_a]
    bars2 = ax2.barh(cat_a, val_a, color=colors_a, height=0.6, alpha=0.9, edgecolor='black', linewidth=0.5)
    ax2.set_xlim(0, 1.25)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xticklabels(['Not Addressed', 'Partial', 'Satisfied'], fontweight='bold', color='#7F8C8D')
    ax2.set_title("Table 5.2: Mapping using Complex Eng. Activities", fontsize=16, fontweight='bold', pad=15)
    ax2.invert_yaxis()
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for bar, val in zip(bars2, val_a):
        lbl = 'Satisfied' if val == 1.0 else 'Partial'
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, lbl, va='center', fontweight='bold', color='#2C3E50', fontsize=12)

    # 5.1 Radar
    ax3 = fig.add_subplot(gs[0, 1], polar=True)
    N = len(cat_p)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    val_p_circ = val_p + val_p[:1]
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(['P1:\nKnowledge', 'P2:\nConflicting', 'P3:\nAnalysis', 'P4:\nFamiliarity', 'P5:\nCodes', 'P6:\nStakeholders', 'P7:\nInterdep.'], 
                        fontsize=10, fontweight='bold', color='#34495E')
    ax3.set_rlabel_position(0)
    ax3.set_yticks([0.5, 1.0])
    ax3.set_yticklabels(["Partial", "Satisfied"], color="#E74C3C", size=9, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.plot(angles, val_p_circ, linewidth=2.5, linestyle='solid', color='#3498DB', marker='o', markersize=8)
    ax3.fill(angles, val_p_circ, '#3498DB', alpha=0.25)
    ax3.set_title("Coverage Area (Table 5.1)", fontsize=14, fontweight='bold', pad=20)

    # 5.2 Radar
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    N2 = len(cat_a)
    angles2 = [n / float(N2) * 2 * np.pi for n in range(N2)]
    angles2 += angles2[:1]
    val_a_circ = val_a + val_a[:1]
    ax4.set_theta_offset(np.pi / 2)
    ax4.set_theta_direction(-1)
    ax4.set_xticks(angles2[:-1])
    ax4.set_xticklabels(['A1:\nResources', 'A2:\nInteraction', 'A3:\nInnovation', 'A4:\nSociety/Env', 'A5:\nFamiliarity'], 
                        fontsize=10, fontweight='bold', color='#34495E')
    ax4.set_rlabel_position(0)
    ax4.set_yticks([0.5, 1.0])
    ax4.set_yticklabels(["Partial", "Satisfied"], color="#E74C3C", size=9, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.plot(angles2, val_a_circ, linewidth=2.5, linestyle='solid', color='#9B59B6', marker='o', markersize=8)
    ax4.fill(angles2, val_a_circ, '#9B59B6', alpha=0.25)
    ax4.set_title("Coverage Area (Table 5.2)", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout(pad=3.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

def create_definitions_infographic(filename):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')

    plt.text(50, 96, "FYDP COMPONENTS GLOSSARY", ha='center', va='center', fontsize=26, fontweight='bold', color='#2C3E50')
    plt.text(50, 92, "Quick reference for terminology found in the Problem Attributes and Outcomes mapping", ha='center', va='center', fontsize=12, style='italic', color='#7F8C8D')

    cols = [
        {"title": "KNOWLEDGE PROFILES (K1-K8)", "color": "#3498DB", "x": 3, "w": 30, "items": [
            "K1: Natural Sciences (Math/Physics used in Ch. 1)",
            "K2: Mathematics (Numerical/Calculus used in Ch. 3/5)",
            "K3: Engineering Fundamentals (Core principles)",
            "K4: Specialist Knowledge (Discipline forefront - Ch. 2/5)",
            "K5: Design Practice (Methods used in Ch. 3)",
            "K6: Engineering Practice (Professional context)",
            "K7: Sustainability / Impact (Society & Env - Ch. 5)",
            "K8: Research Literature (Peer-reviewed - Ch. 2)"
        ]},
        {"title": "PROGRAM OUTCOMES (PO)", "color": "#2ECC71", "x": 35, "w": 30, "items": [
            "PO1: Engineering Knowledge (Theory/Science - Ch. 2)",
            "PO2: Problem Analysis (First principles - Ch. 1)",
            "PO4: Investigation (Data & conclusions - Ch. 5)",
            "PO10: Communication (Writing/Formatting - ALL)",
            "PO11: Project Management (Timeline/Budget - Ch. 3)",
            "PO12: Life-long Learning (Self-learning - Ch. 5)"
        ]},
        {"title": "PROBLEM ATTRIBUTES (A1-A7)", "color": "#E74C3C", "x": 67, "w": 30, "items": [
            "A1: Depth of Knowledge (First principles/Theory)",
            "A2: Conflicting Requirements (Trade-offs)",
            "A3: Depth of Analysis (Formal modeling/Simulation)",
            "A4: Familiarity of Issues (Non-routine problems)",
            "A5: Applicable Codes (IEEE/ISO adaptation)",
            "A6: Stakeholder Involvement (Users, Govt, etc.)",
            "A7: Interdependence (Changing 1 affects many)"
        ]}
    ]
    
    for col in cols:
        # Header box
        rect = mpatches.FancyBboxPatch((col["x"], 85), col["w"], 4.5, boxstyle="round,pad=1,rounding_size=1", facecolor=col["color"], edgecolor='none')
        ax.add_patch(rect)
        ax.text(col["x"] + col["w"]/2, 87.25, col["title"], ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        y_pos = 79
        for item in col["items"]:
            # item box
            rect_i = mpatches.FancyBboxPatch((col["x"], y_pos-3.5), col["w"], 7, boxstyle="round,pad=0.5,rounding_size=0.5", facecolor='white', edgecolor='#BDC3C7', lw=1.5)
            ax.add_patch(rect_i)
            
            prefix, desc = item.split(": ", 1)
            ax.text(col["x"]+1.5, y_pos+1.5, prefix + ":", fontweight='bold', fontsize=12, color='#2C3E50', va='center')
            
            wrapped = textwrap.fill(desc, width=38)
            ax.text(col["x"]+1.5, y_pos-1.5, wrapped, fontsize=11, color='#34495E', va='center')
            
            y_pos -= 9.5

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

if __name__ == '__main__':
    create_chapter_mapping_infographic('d:/FYDP/POCD-KansformerEPI/Vis_1_Chapter_Mapping.png')
    create_combined_tables_infographic('d:/FYDP/POCD-KansformerEPI/Vis_2_Combined_Tables.png')
    create_definitions_infographic('d:/FYDP/POCD-KansformerEPI/Vis_3_Glossary.png')
    print("All visual charts successfully generated!")
