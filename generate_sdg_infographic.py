import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

# Set global styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'Helvetica', 'sans-serif']

def create_sdg_infographic(filename):
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    # Background intentionally left transparent

    sdgs = [
        {"id": "SDG 3", "title": "Good Health\n& Well-Being", 
         "desc": "Enables faster gene-regulation discovery by predicting enhancer-promoter interactions to support biomedical research.",
         "color": "#4CAF50"},  # Green
        {"id": "SDG 4", "title": "Quality\nEducation", 
         "desc": "Demonstrates reproducible, modern genomics ML methods (encoding, attention, evaluation, interpretability) for learning and training.",
         "color": "#C0392B"},  # Red
        {"id": "SDG 9", "title": "Industry,\nInnovation\n& Infrastructure", 
         "desc": "Builds a scalable, interpretable AI pipeline that strengthens bioinformatics/biotech research infrastructure.",
         "color": "#F39C12"},  # Orange
        {"id": "SDG 14", "title": "Responsible\nConsumption\n& Production", 
         "desc": "Reduces costly wet-lab trial-and-error by prioritizing the most likely regulatory interactions for validation.",
         "color": "#2980B9"},  # Blue
        {"id": "SDG 17", "title": "Partnerships\nfor the Goals", 
         "desc": "Leverages shared datasets (e.g., ENCODE) and interpretable outputs to improve collaboration between ML and biology teams.",
         "color": "#1A252F"}   # Navy
    ]

    card_width = 17
    card_height = 65
    spacing = 2
    
    # Calculate starting x position to center cards horizontally
    total_width = (len(sdgs) * card_width) + ((len(sdgs) - 1) * spacing)
    x_pos = (100 - total_width) / 2

    for sdg in sdgs:
        # Card Background - Set to the SDG theme color
        rect_card = mpatches.FancyBboxPatch((x_pos, 10), card_width, card_height, boxstyle="round,pad=1,rounding_size=2", facecolor=sdg["color"], edgecolor='none')
        ax.add_patch(rect_card)

        # Top Color Banner (for ID/Number) - slightly darker shade or keep same for unified look
        # We can keep it the same color with an outline, or just white text at the top
        rect_banner = mpatches.FancyBboxPatch((x_pos, 75 - 12), card_width, 12, boxstyle="round,pad=0,rounding_size=2", facecolor=sdg["color"], edgecolor='none')
        # Square off the bottom corners of the banner so it sits flush inside the card
        rect_square_base = mpatches.Rectangle((x_pos, 75 - 12), card_width, 8, facecolor=sdg["color"], edgecolor='none')
        
        ax.add_patch(rect_banner)
        ax.add_patch(rect_square_base)
        
        # ID text in banner (White)
        ax.text(x_pos + card_width/2, 69, sdg["id"], ha='center', va='center', fontsize=26, fontweight='bold', color='white')

        # SDG Title text (White)
        ax.text(x_pos + card_width/2, 54, sdg["title"], ha='center', va='center', fontsize=20, fontweight='bold', color='white', linespacing=1.2)
        
        # Divider Line (White with low alpha)
        ax.plot([x_pos + 2, x_pos + card_width - 2], [43, 43], color='white', alpha=0.5, lw=2)

        # Description text (Off-white for contrast)
        wrapped_desc = textwrap.fill(sdg["desc"], width=20)
        ax.text(x_pos + card_width/2, 38, wrapped_desc, ha='center', va='top', fontsize=16, color='#F8F9FA', linespacing=1.4)

        x_pos += card_width + spacing

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    create_sdg_infographic('d:/FYDP/POCD-KansformerEPI/Vis_4_SDG_Mapping.png')
    print("SDG Mapping Infographic generated successfully!")
