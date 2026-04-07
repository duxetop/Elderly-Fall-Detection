"""Generate system architecture diagram for poster (31x21 inches)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
C_ROOM    = '#E8F0FE'  # light blue bg
C_PI      = '#1A73E8'  # blue
C_PERIPH  = '#34A853'  # green
C_SERVER  = '#FBBC04'  # yellow
C_PHONE   = '#EA4335'  # red
C_COMP    = '#F1F3F4'  # component grey
C_TEXT    = '#202124'
C_BORDER  = '#5F6368'

# ── Room Unit box ──
room = FancyBboxPatch((0.5, 2.8), 11.5, 6.8, boxstyle="round,pad=0.2",
                       facecolor=C_ROOM, edgecolor=C_BORDER, linewidth=2)
ax.add_patch(room)
ax.text(6.25, 9.25, 'Room Unit', fontsize=18, fontweight='bold',
        color=C_TEXT, ha='center', va='center')

# ── Camera ──
cam = FancyBboxPatch((1.2, 7.2), 3.0, 1.8, boxstyle="round,pad=0.15",
                      facecolor=C_COMP, edgecolor=C_BORDER, linewidth=1.5)
ax.add_patch(cam)
ax.text(2.7, 8.35, 'Camera', fontsize=13, fontweight='bold', ha='center', color=C_TEXT)
ax.text(2.7, 7.75, '320x240 @ 23 FPS', fontsize=9, ha='center', color='#5F6368')

# ── IMU ──
imu = FancyBboxPatch((1.2, 4.6), 3.0, 1.8, boxstyle="round,pad=0.15",
                      facecolor=C_COMP, edgecolor=C_BORDER, linewidth=1.5)
ax.add_patch(imu)
ax.text(2.7, 5.75, 'IMU Sensor', fontsize=13, fontweight='bold', ha='center', color=C_TEXT)
ax.text(2.7, 5.15, 'ICM-42670-P\n6-axis accel/gyro', fontsize=9, ha='center', color='#5F6368')

# ── Pi 4 (main) ──
pi = FancyBboxPatch((5.5, 4.2), 4.8, 5.2, boxstyle="round,pad=0.2",
                     facecolor='white', edgecolor=C_PI, linewidth=2.5)
ax.add_patch(pi)
ax.text(7.9, 9.0, 'Raspberry Pi 4', fontsize=15, fontweight='bold',
        ha='center', color=C_PI)

# Sub-boxes inside Pi
def pi_box(x, y, w, h, title, detail, color=C_PI):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                        facecolor=color + '18', edgecolor=color, linewidth=1.2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h*0.65, title, fontsize=10, fontweight='bold',
            ha='center', color=color)
    ax.text(x + w/2, y + h*0.25, detail, fontsize=7.5, ha='center', color='#5F6368')

pi_box(5.8, 7.6, 4.2, 1.1, 'MoveNet Lightning INT8', '17 keypoints  |  19ms inference', C_PI)
pi_box(5.8, 6.2, 4.2, 1.1, 'Fall Score Engine', '8 signals  |  threshold 0.45', '#E8710A')
pi_box(5.8, 4.6, 2.0, 1.3, 'State\nMachine', 'unknown\npotential\nfallen', '#7B1FA2')
pi_box(8.0, 4.6, 2.0, 1.3, 'Camera\nCalibration', 'auto tilt\ncorrection', '#0D652D')

# ── Peripheral Controller ──
periph = FancyBboxPatch((1.0, 3.0), 3.2, 1.3, boxstyle="round,pad=0.15",
                         facecolor=C_PERIPH + '20', edgecolor=C_PERIPH, linewidth=2)
ax.add_patch(periph)
ax.text(2.6, 3.95, 'Peripheral Controller', fontsize=11, fontweight='bold',
        ha='center', color=C_PERIPH)
ax.text(2.6, 3.35, 'Buzzer  |  RGB LED  |  Mic  |  Speaker',
        fontsize=8, ha='center', color='#5F6368')

# ── Relay Server ──
server = FancyBboxPatch((6.0, 1.0), 3.8, 1.2, boxstyle="round,pad=0.15",
                         facecolor=C_SERVER + '30', edgecolor=C_SERVER, linewidth=2)
ax.add_patch(server)
ax.text(7.9, 1.6, 'WebSocket Relay Server', fontsize=12, fontweight='bold',
        ha='center', color='#E37400')

# ── Caregiver App ──
phone = FancyBboxPatch((12.5, 3.0), 3.0, 6.2, boxstyle="round,pad=0.2",
                        facecolor=C_PHONE + '15', edgecolor=C_PHONE, linewidth=2.5)
ax.add_patch(phone)
ax.text(14.0, 8.7, 'Caregiver', fontsize=15, fontweight='bold', ha='center', color=C_PHONE)
ax.text(14.0, 8.15, 'Mobile App', fontsize=15, fontweight='bold', ha='center', color=C_PHONE)

# Phone items
phone_items = [
    ('Fall Alerts', 7.2),
    ('Scene Snapshots', 6.5),
    ('IMU Data Upload', 5.8),
    ('Alert Cancel', 5.1),
    ('Voice Commands', 4.4),
]
for label, ypos in phone_items:
    pill = FancyBboxPatch((12.9, ypos - 0.22), 2.2, 0.5, boxstyle="round,pad=0.08",
                           facecolor='white', edgecolor=C_PHONE, linewidth=1)
    ax.add_patch(pill)
    ax.text(14.0, ypos + 0.03, label, fontsize=9, ha='center', color=C_TEXT)

# ── Arrows ──
arrow_kw = dict(arrowstyle='->', mutation_scale=18, linewidth=2.2)
arrow_kw_bi = dict(arrowstyle='<->', mutation_scale=18, linewidth=2.2)

# Camera → Pi
ax.annotate('', xy=(5.5, 8.1), xytext=(4.2, 8.1),
            arrowprops=dict(**arrow_kw, color=C_BORDER))
ax.text(4.85, 8.45, 'keypoints', fontsize=8, ha='center', color='#5F6368', style='italic')

# IMU → Pi
ax.annotate('', xy=(5.5, 5.5), xytext=(4.2, 5.5),
            arrowprops=dict(**arrow_kw, color=C_BORDER))
ax.text(4.85, 5.85, 'accel/gyro', fontsize=8, ha='center', color='#5F6368', style='italic')

# Pi ↔ Peripheral Controller (IPC)
ax.annotate('', xy=(4.2, 3.65), xytext=(5.5, 4.8),
            arrowprops=dict(**arrow_kw_bi, color=C_PERIPH, connectionstyle='arc3,rad=0.2'))
ax.text(4.2, 4.5, 'IPC\nsocket', fontsize=8, ha='center', color=C_PERIPH, fontweight='bold')

# Pi → Server (WebSocket down)
ax.annotate('', xy=(7.9, 2.2), xytext=(7.9, 4.2),
            arrowprops=dict(**arrow_kw_bi, color=C_SERVER, connectionstyle='arc3,rad=0'))
ax.text(8.6, 3.2, 'WebSocket', fontsize=9, ha='center', color='#E37400', fontweight='bold')

# Server → Phone
ax.annotate('', xy=(12.5, 4.5), xytext=(9.8, 1.6),
            arrowprops=dict(**arrow_kw_bi, color=C_PHONE, connectionstyle='arc3,rad=-0.3'))

# ── Legend ──
legend_items = [
    (C_PI, 'Pose Estimation'),
    (C_PERIPH, 'Peripheral Control'),
    (C_SERVER, 'Network Relay'),
    (C_PHONE, 'Mobile App'),
]
for i, (color, label) in enumerate(legend_items):
    yy = 1.8 - i * 0.35
    rect = FancyBboxPatch((0.6, yy - 0.12), 0.3, 0.25, boxstyle="round,pad=0.03",
                           facecolor=color, edgecolor='none')
    ax.add_patch(rect)
    ax.text(1.1, yy, label, fontsize=9, va='center', color=C_TEXT)

plt.tight_layout(pad=0.5)
plt.savefig('/home/htanoli/Desktop/ElderlyFall/system_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: system_architecture.png")
