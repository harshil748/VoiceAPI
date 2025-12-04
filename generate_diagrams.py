"""Generate system architecture diagrams as PNG files"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

plt.style.use('default')
plt.rcParams['font.size'] = 10

def create_system_architecture():
    """Main system architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('VoiceAPI - System Architecture', fontsize=20, fontweight='bold', pad=20)
    
    colors = {
        'client': '#E3F2FD', 'api': '#FFF3E0', 'engine': '#E8F5E9',
        'model': '#FCE4EC', 'output': '#F3E5F5', 'lang': '#FFFDE7'
    }
    
    # Client Applications
    ax.add_patch(FancyBboxPatch((0.5, 9), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor=colors['client'], edgecolor='#1976D2', linewidth=2))
    ax.text(2.5, 11.2, 'CLIENT APPLICATIONS', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 10.5, '* Web App', ha='center', fontsize=10)
    ax.text(2.5, 10, '* Mobile App', ha='center', fontsize=10)
    ax.text(2.5, 9.5, '* Healthcare Assistant', ha='center', fontsize=10)
    
    # API Server
    ax.add_patch(FancyBboxPatch((5.5, 9), 5, 2.5, boxstyle="round,pad=0.1",
                                facecolor=colors['api'], edgecolor='#F57C00', linewidth=2))
    ax.text(8, 11.2, 'FastAPI SERVER (Port 7860)', ha='center', fontsize=12, fontweight='bold')
    ax.text(8, 10.5, '/Get_Inference API', ha='center', fontsize=11, style='italic')
    ax.text(8, 10, 'Params: text, lang, speaker_wav', ha='center', fontsize=9)
    ax.text(8, 9.5, 'Response: audio/wav', ha='center', fontsize=9)
    
    # TTS Engine
    ax.add_patch(FancyBboxPatch((0.5, 4), 10, 4.5, boxstyle="round,pad=0.1",
                                facecolor=colors['engine'], edgecolor='#388E3C', linewidth=2))
    ax.text(5.5, 8.2, 'TTS ENGINE', ha='center', fontsize=14, fontweight='bold')
    
    # Engine components
    for i, (name, x) in enumerate([('Text Normalizer', 2.25), ('Tokenizer', 5.25), ('Style Processor', 8.5)]):
        ax.add_patch(FancyBboxPatch((x-1.25, 6.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                                    facecolor='white', edgecolor='#666'))
        ax.text(x, 7.1, name, ha='center', fontsize=9, fontweight='bold')
    
    # Models
    ax.add_patch(FancyBboxPatch((1, 4.3), 9, 1.8, boxstyle="round,pad=0.05",
                                facecolor=colors['model'], edgecolor='#C2185B', linewidth=1.5))
    ax.text(5.5, 5.8, 'MODEL TYPES', ha='center', fontsize=11, fontweight='bold')
    for i, (name, x) in enumerate([('VITS JIT (.pt)', 2.5), ('Coqui TTS (.pth)', 5.5), ('Facebook MMS', 8.5)]):
        ax.text(x, 5.1, name, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    # Languages
    ax.add_patch(FancyBboxPatch((11.5, 4), 4, 7.5, boxstyle="round,pad=0.1",
                                facecolor=colors['lang'], edgecolor='#FBC02D', linewidth=2))
    ax.text(13.5, 11.2, '11 LANGUAGES', ha='center', fontsize=12, fontweight='bold')
    languages = ['Hindi', 'Bengali', 'Marathi', 'Telugu', 'Kannada', 'Gujarati',
                 'Bhojpuri', 'Chhattisgarhi', 'Maithili', 'Magahi', 'English']
    for i, lang in enumerate(languages):
        ax.text(13.5, 10.5 - i*0.6, lang, ha='center', fontsize=9)
    
    # Output
    ax.add_patch(FancyBboxPatch((5.5, 0.5), 5, 2.5, boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='#7B1FA2', linewidth=2))
    ax.text(8, 2.7, 'AUDIO OUTPUT', ha='center', fontsize=12, fontweight='bold')
    ax.text(8, 2, 'WAV File @ 22050 Hz', ha='center', fontsize=11)
    ax.text(8, 1.4, '(16000 Hz for Gujarati)', ha='center', fontsize=9, style='italic')
    
    # Arrows
    ax.annotate('', xy=(5.5, 10.25), xytext=(4.5, 10.25),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))
    ax.annotate('', xy=(8, 9), xytext=(8, 8.5),
                arrowprops=dict(arrowstyle='->', color='#F57C00', lw=2))
    ax.annotate('', xy=(8, 4), xytext=(8, 3),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=2))
    ax.annotate('', xy=(11.5, 6), xytext=(10.5, 6),
                arrowprops=dict(arrowstyle='<->', color='#666', lw=1.5))
    
    ax.text(5, 10.5, 'HTTP', fontsize=8, color='#1976D2')
    ax.text(8.2, 8.7, 'text, lang', fontsize=8, color='#F57C00')
    ax.text(8.2, 3.4, 'audio', fontsize=8, color='#388E3C')
    
    plt.tight_layout()
    plt.savefig('diagrams/system_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: diagrams/system_architecture.png")

def create_data_flow():
    """Data flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('VoiceAPI - Data Flow Diagram', fontsize=18, fontweight='bold', pad=20)
    
    components = [('Client', 1.5, '#E3F2FD'), ('API Server', 4.5, '#FFF3E0'),
                  ('TTS Engine', 7.5, '#E8F5E9'), ('Model', 10.5, '#FCE4EC')]
    
    for name, x, color in components:
        ax.add_patch(FancyBboxPatch((x-0.8, 9), 1.6, 0.8, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black', linewidth=1.5))
        ax.text(x, 9.4, name, ha='center', fontsize=10, fontweight='bold')
        ax.plot([x, x], [0.5, 9], 'k--', linewidth=1, alpha=0.5)
    
    messages = [
        (1.5, 4.5, 8.5, '1. GET /Get_Inference?text=...&lang=hindi', '#1976D2'),
        (4.5, 7.5, 7.8, '2. synthesize(text, voice)', '#F57C00'),
        (7.5, 7.5, 7.3, '3. Normalize & Tokenize', '#388E3C'),
        (7.5, 10.5, 6.5, '4. Load model', '#C2185B'),
        (10.5, 10.5, 6.0, '5. Forward pass (inference)', '#C2185B'),
        (10.5, 7.5, 5.3, '6. Raw audio tensor', '#C2185B'),
        (7.5, 7.5, 4.8, '7. Apply style (pitch/speed)', '#388E3C'),
        (7.5, 4.5, 4.0, '8. TTSOutput', '#F57C00'),
        (4.5, 4.5, 3.5, '9. Convert to WAV bytes', '#F57C00'),
        (4.5, 1.5, 2.5, '10. audio/wav response', '#1976D2'),
    ]
    
    for x1, x2, y, label, color in messages:
        if x1 < x2:
            ax.annotate('', xy=(x2-0.1, y), xytext=(x1+0.1, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        elif x1 > x2:
            ax.annotate('', xy=(x2+0.1, y), xytext=(x1-0.1, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        else:
            ax.annotate('', xy=(x1+0.3, y-0.3), xytext=(x1+0.3, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                      connectionstyle="arc3,rad=0.3"))
        text_x = (x1 + x2) / 2 if x1 != x2 else x1 + 1.5
        ax.text(text_x, y + 0.15, label, ha='center', fontsize=8, color=color)
    
    plt.tight_layout()
    plt.savefig('diagrams/data_flow.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: diagrams/data_flow.png")

def create_model_architecture():
    """VITS model architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('VITS Model Architecture', fontsize=18, fontweight='bold', pad=20)
    
    blocks = [
        ('INPUT\n\nText\n"Hello"', 0.5, '#E3F2FD', '#1976D2'),
        ('TEXT ENCODER\n\nChar Embedding\n     |\nTransformer\n(6 layers, 192 hidden)', 3.5, '#FFF3E0', '#F57C00'),
        ('FLOW MODEL\n\nPrior Encoder\n     |\nNormalizing Flow\n     |\nDuration Predictor', 7.5, '#E8F5E9', '#388E3C'),
        ('HiFi-GAN DECODER\n\nUpsampling\n     |\nResidual Blocks\n     |\nAudio Output', 11.5, '#FCE4EC', '#C2185B'),
    ]
    
    for text, x, fc, ec in blocks:
        ax.add_patch(FancyBboxPatch((x, 2.5), 3, 3.5, boxstyle="round,pad=0.1",
                                    facecolor=fc, edgecolor=ec, linewidth=2))
        ax.text(x+1.5, 4.25, text, ha='center', va='center', fontsize=9, linespacing=1.3)
    
    # Arrows
    for x in [3.4, 7.4, 11.4]:
        ax.annotate('', xy=(x+0.1, 4.25), xytext=(x-0.4, 4.25),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Waveform
    ax.add_patch(FancyBboxPatch((14.8, 3.5), 1, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5))
    x_wave = np.linspace(15, 15.6, 50)
    y_wave = 4.25 + 0.3 * np.sin(np.linspace(0, 8*np.pi, 50))
    ax.plot(x_wave, y_wave, color='#7B1FA2', linewidth=1.5)
    ax.annotate('', xy=(14.7, 4.25), xytext=(14.6, 4.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.text(8, 1.5, 'Sample Rate: 22050 Hz  |  Duration: Stochastic  |  Vocoder: HiFi-GAN',
            ha='center', fontsize=10, style='italic', color='#666')
    
    plt.tight_layout()
    plt.savefig('diagrams/model_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: diagrams/model_architecture.png")

def create_training_pipeline():
    """Training pipeline"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('VoiceAPI - Training Pipeline', fontsize=18, fontweight='bold', pad=20)
    
    # Data Sources
    ax.add_patch(FancyBboxPatch((0.5, 7.5), 4, 2, boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(2.5, 9.2, 'TRAINING DATA', ha='center', fontsize=12, fontweight='bold')
    for i, ds in enumerate(['OpenSLR (Hindi, Bengali...)', 'Mozilla Common Voice',
                           'IndicTTS Corpus', 'AI4Bharat Indic-Voices']):
        ax.text(2.5, 8.6 - i*0.35, '* ' + ds, ha='center', fontsize=8)
    
    # Data Prep
    ax.add_patch(FancyBboxPatch((5.5, 7.5), 3.5, 2, boxstyle="round,pad=0.1",
                                facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2))
    ax.text(7.25, 9.2, 'DATA PREP', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.25, 8.6, 'prepare_dataset.py', ha='center', fontsize=9, style='italic')
    ax.text(7.25, 8.1, '* Normalize to 22050 Hz', ha='center', fontsize=8)
    ax.text(7.25, 7.7, '* Generate transcripts', ha='center', fontsize=8)
    
    # Training
    ax.add_patch(FancyBboxPatch((2, 4), 5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(4.5, 6.2, 'TRAINING', ha='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 5.7, 'train_vits.py', ha='center', fontsize=9, style='italic')
    for i, t in enumerate(['* 1000 epochs', '* Batch size: 32', '* AdamW optimizer']):
        ax.text(4.5, 5.2 - i*0.35, t, ha='center', fontsize=8)
    
    # Config
    ax.add_patch(FancyBboxPatch((8, 4.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FFFDE7', edgecolor='#FBC02D', linewidth=2))
    ax.text(9.25, 5.7, 'CONFIG', ha='center', fontsize=11, fontweight='bold')
    ax.text(9.25, 5.2, 'configs/*.yaml', ha='center', fontsize=9, style='italic')
    ax.text(9.25, 4.8, 'hindi_female.yaml', ha='center', fontsize=8)
    
    # Export
    ax.add_patch(FancyBboxPatch((2, 0.5), 5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2))
    ax.text(4.5, 2.7, 'EXPORT', ha='center', fontsize=12, fontweight='bold')
    ax.text(4.5, 2.2, 'export_model.py', ha='center', fontsize=9, style='italic')
    for i, t in enumerate(['* JIT trace -> .pt', '* Generate chars.txt', '* Package for inference']):
        ax.text(4.5, 1.7 - i*0.35, t, ha='center', fontsize=8)
    
    # Output
    ax.add_patch(FancyBboxPatch((8, 0.5), 4, 2.5, boxstyle="round,pad=0.1",
                                facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2))
    ax.text(10, 2.7, 'OUTPUT', ha='center', fontsize=12, fontweight='bold')
    ax.text(10, 2.1, 'models/', ha='center', fontsize=10, fontweight='bold')
    ax.text(8.5, 1.5, 'hi_male/  hi_female/', ha='left', fontsize=8)
    ax.text(8.5, 1.1, 'bn_male/  bn_female/', ha='left', fontsize=8)
    ax.text(8.5, 0.7, '...21 voices total', ha='left', fontsize=8, style='italic')
    
    # Arrows
    ax.annotate('', xy=(5.4, 8.5), xytext=(4.6, 8.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(4.5, 7.4), xytext=(7.25, 7.4), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(7.9, 5.25), xytext=(7.1, 5.25), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(4.5, 4), xytext=(4.5, 3.1), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(7.9, 1.75), xytext=(7.1, 1.75), arrowprops=dict(arrowstyle='->', lw=2))
    
    plt.tight_layout()
    plt.savefig('diagrams/training_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: diagrams/training_pipeline.png")

def create_voice_map():
    """Voice configuration map"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    ax.set_title('VoiceAPI - 21 Voices across 11 Languages', fontsize=18, fontweight='bold', pad=20)
    
    languages = [
        ('Hindi', ['hi_male', 'hi_female'], '#E3F2FD'),
        ('Bengali', ['bn_male', 'bn_female'], '#FFF3E0'),
        ('Marathi', ['mr_male', 'mr_female'], '#FCE4EC'),
        ('Telugu', ['te_male', 'te_female'], '#E8F5E9'),
        ('Kannada', ['kn_male', 'kn_female'], '#F3E5F5'),
        ('Gujarati', ['gu_mms'], '#FFFDE7'),
        ('Bhojpuri', ['bho_male', 'bho_female'], '#E0F7FA'),
        ('Chhattisgarhi', ['hne_male', 'hne_female'], '#FBE9E7'),
        ('Maithili', ['mai_male', 'mai_female'], '#F1F8E9'),
        ('Magahi', ['mag_male', 'mag_female'], '#EDE7F6'),
        ('English', ['en_male', 'en_female'], '#E3F2FD'),
    ]
    
    for i, (lang, voices, color) in enumerate(languages):
        y = 10 - i * 0.9
        # Language box
        ax.add_patch(FancyBboxPatch((0.5, y-0.35), 2.5, 0.7, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='#666', linewidth=1))
        ax.text(1.75, y, lang, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Voice boxes
        for j, voice in enumerate(voices):
            ax.add_patch(FancyBboxPatch((3.5 + j*3, y-0.3), 2.7, 0.6, boxstyle="round,pad=0.03",
                                        facecolor='white', edgecolor='#999', linewidth=1))
            gender = '(M)' if 'male' in voice else '(F)'
            ax.text(3.5 + j*3 + 1.35, y, f'{voice} {gender}', ha='center', va='center', fontsize=9)
        
        ax.plot([3, 3.5], [y, y], 'k-', linewidth=1, alpha=0.5)
    
    # Stats
    ax.add_patch(FancyBboxPatch((10, 8.5), 3.5, 2, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(11.75, 10.2, 'STATS', ha='center', fontsize=12, fontweight='bold')
    ax.text(11.75, 9.5, '11 Languages', ha='center', fontsize=10)
    ax.text(11.75, 9, '21 Voices', ha='center', fontsize=10)
    ax.text(11.75, 8.5, '~8GB Models', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('diagrams/voice_map.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: diagrams/voice_map.png")

if __name__ == '__main__':
    print("Generating VoiceAPI diagrams...\n")
    create_system_architecture()
    create_data_flow()
    create_model_architecture()
    create_training_pipeline()
    create_voice_map()
    print("\nAll diagrams saved to diagrams/ folder!")
