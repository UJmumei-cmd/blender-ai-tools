"""基準の数字を1か所にまとめる（CLAUDE.md 7章）。

- 長さはメートルで持つ（Blender の中の単位）。読みやすいように mm() で書く。
- 「仮」と書いた数字は、まだ決まっていない。Mac mini で本物のモデルを測ってから決め直す。
- 出所がある数字は、出所（ロードマップの番号・convert_blend.py など）を書く。
"""


def mm(value: float) -> float:
    """ミリメートルをメートルにする"""
    return value / 1000.0


# MARK: - 書き出し（E1・E2）

# 1つの頂点に効く骨の数の上限（convert_blend.py の MAX_BONE_INFLUENCES と同じ。iPhone の RealityKit 向け）
MAX_BONE_INFLUENCES = 4
# 読み戻した形のずれの上限（ロードマップ段階1 基準4）
ROUNDTRIP_MAX_DEVIATION = mm(0.5)

# MARK: - 再現性（A15）

# 同じ入力を2回測ったときに許す差。これより小さい差は、変化として数えない（仮）
REPRO_TOLERANCE_LENGTH = mm(0.01)
REPRO_TOLERANCE_ANGLE_DEG = 0.01
REPRO_TOLERANCE_RATIO = 1e-6
# 乱数の種を指定しなかったときの種
DEFAULT_SEED = 0

# MARK: - ウェイト（A4・B4）

# これより小さい重みは 0 とみなす（仮）
WEIGHT_EPSILON = 1e-6
# 重みの合計が 1 からずれてよい幅（仮）
WEIGHT_SUM_TOLERANCE = 1e-3

# MARK: - めり込み・すき間（A4）

# 体が服から出る・服どうしが突き抜ける深さ（仮）
PENETRATION_YELLOW = mm(0.5)
PENETRATION_RED = mm(1.0)
# 服と体のすき間（ロードマップ段階3 R2）
CLOTH_GAP_MIN = mm(1.0)
CLOTH_GAP_MAX = mm(4.0)

# MARK: - 左右のずれ（A4・B11）

# 左右を映したときの、いちばん近い点までの距離（仮）
ASYMMETRY_YELLOW = mm(1.0)
ASYMMETRY_RED = mm(3.0)

# MARK: - 形のくずれ（A4）

# 辺が休みの形の何倍まで伸びてよいか（仮）
EDGE_STRETCH_YELLOW = 1.3
EDGE_STRETCH_RED = 1.6
# 辺が休みの形の何倍まで縮んでよいか（仮）
EDGE_SHRINK_YELLOW = 0.7
EDGE_SHRINK_RED = 0.5
# 体積が減ってよい割合（仮）
VOLUME_LOSS_YELLOW = 0.10
VOLUME_LOSS_RED = 0.25
# とげの強さ：まわりの辺の長さの平均に対する、とび出しの割合（仮）
SPIKE_YELLOW = 0.5
SPIKE_RED = 1.0

# MARK: - 面の点検（安全な直し）

# 重なった点とみなす距離（Blender の「距離でマージ」の既定 0.0001 m と同じ）
MERGE_DISTANCE = mm(0.1)
# 面積 0 とみなす面積（平方メートル）（仮）
DEGENERATE_AREA = 1e-12

# MARK: - 関節と足（A4・M4）

# ひざ・ひじの逆曲がりとみなす角度（仮）
REVERSE_BEND_TOLERANCE_DEG = 2.0
# 足が床に沈んだとみなす深さ（仮）
FLOOR_SINK_TOLERANCE = mm(2.0)
# 足が床に着いているとみなす高さと速さ（仮）
FOOT_CONTACT_HEIGHT = mm(20.0)
FOOT_CONTACT_SPEED = 0.1  # m/s
# 着いている間に滑ってよい距離（仮）
FOOT_SLIDE_TOLERANCE = mm(5.0)
# 目標までの距離が鎖の長さをこの割合より超えたら「届かない」（仮）
IK_REACH_TOLERANCE = 0.01
# 1フレームで回ってよい角度。これを超えたら「跳ぶ」（仮）
BONE_JUMP_DEG = 30.0

# MARK: - 見える・見えない（A4）

# 見えるかどうかを調べる画像の大きさ（ピクセル、長い辺）（仮）
VISIBILITY_RESOLUTION = 512
# 奥行きの比べで、同じ面とみなす幅（仮）
VISIBILITY_DEPTH_EPSILON = mm(1.0)
# アプリのカメラの画角（KICKOFF の「画角70°」。縦か横かは、紅魔館のコードで確かめる）
APP_CAMERA_FOV_DEG = 70.0
