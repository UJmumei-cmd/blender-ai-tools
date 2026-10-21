"""基準の数字を1か所にまとめる（CLAUDE.md 7章）。

- 長さはメートルで持つ（Blender の中の単位）。読みやすいように mm() で書く。
- 「仮」と書いた数字は、まだ決まっていない。Mac mini で本物のモデルを測ってから決め直す。
- 出所がある数字は、出所（ロードマップの番号・convert_blend.py など）を書く。
"""


def mm(value: float) -> float:
    """ミリメートルをメートルにする"""
    return value / 1000.0


# MARK: - 道具の目録（A8）

# LLM の返事（道具の呼び出しの JSON）の長さの上限（文字）。parse_tool_call の max_length の既定（仮）
MAX_TOOL_CALL_LENGTH = 65536
# 道具の入力のスキーマの入れ子の深さの上限（いちばん外が 0 段）。深すぎるスキーマで止まらなくならないようにする（仮）
MAX_SCHEMA_DEPTH = 32
# 骨の名前の対応（bones.map_names）で1回に受け取る名前の数と、1つの名前の長さ（文字）の上限（仮。揺れ物の骨が多いモデルでも足りるように多めにする）
MAX_BONE_NAMES = 10000
MAX_BONE_NAME_LENGTH = 256

# MARK: - 書き出し（E1・E2）

# 1つの頂点に効く骨の数の上限（convert_blend.py の MAX_BONE_INFLUENCES と同じ。iPhone の RealityKit 向け）
MAX_BONE_INFLUENCES = 4
# 読み戻した形のずれの上限（ロードマップ段階1 基準4）
ROUNDTRIP_MAX_DEVIATION = mm(0.5)
# 往復の確認で、書き出した点と元の点を「同じ点」とみなす距離（休みの姿勢。float32 の誤差より大きく、0.5mm よりずっと小さい）（仮）
ROUNDTRIP_MATCH_DISTANCE = mm(0.01)
# 往復の確認で、書き出した点と元の頂点を並びのまま対応させるとき、ROUNDTRIP_MATCH_DISTANCE の中に入る点がこの割合以上なら、
# 並びが合っているとみなす（割合。合わなければ、いちばん近い点で対応させる。そちらのほうが多く合うときだけ）（仮）
ORDER_MATCH_MIN_FRACTION = 0.5
# 往復の確認で、restTransforms から組み立てた休みの姿勢と bindTransforms の、行列の成分の差の上限（仮）。
# 骨格の空間の数（メートルではない。回転は約 1e-5 rad、位置は大きさ 1 のアーマチュアで 0.01mm）。
# これより大きい骨を「合わない骨」として出す（形が動かなければ黄）
SKELETON_MATRIX_TOLERANCE = 1e-5
# 往復の確認で、ポーズの骨が「動いた」とみなす、休みの姿勢との行列の成分の差（骨格の空間の数）（仮）。
# どの骨も動かなければ、ポーズを確かめられなかったので灰にする（IK で戻された など）
POSE_MOVED_TOLERANCE = 1e-6
# convert_blend.py を待つ時間の上限（秒）。過ぎたら止めて、失敗として報告する（仮）
CONVERT_TIMEOUT_SECONDS = 600.0

# MARK: - 裏で動かす係（A10）

# 進み具合を読みに行く間隔（秒）。画面のタイマーで、裏の Blender が書く progress.json を読む（仮）
JOB_POLL_INTERVAL_SECONDS = 1.0
# 中止を頼んでから、裏の Blender が自分で終わるのを待つ時間（秒）。過ぎたら止める（terminate → さらに同じ時間で kill）（仮）
JOB_CANCEL_GRACE_SECONDS = 10.0
# 1つの仕事を待つ時間の上限（秒）。過ぎたら中止と同じ手順で止め、失敗として記録する（仮。夜のじっくり点検では長くする）
JOB_TIMEOUT_SECONDS = 4.0 * 3600.0

# MARK: - 再現性（A15）

# 同じ入力を2回測ったときに許す差。これより小さい差は、変化として数えない（仮）
REPRO_TOLERANCE_LENGTH = mm(0.01)
REPRO_TOLERANCE_ANGLE_DEG = 0.01
REPRO_TOLERANCE_RATIO = 1e-6
# 同じ入力を2回測ったときに許すウェイト（重み）の差。これより小さい差は、変化として数えない（仮）。
# WEIGHT_EPSILON（0 とみなす重み）とは意味が違うので、別に持つ
REPRO_TOLERANCE_WEIGHT = 1e-6
# 乱数の種を指定しなかったときの種
DEFAULT_SEED = 0
# git の答えを待つ時間の上限（秒）。git_state の1回の git ごと。過ぎたら「分からない」（None）にする（仮）
GIT_TIMEOUT_SECONDS = 10.0

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
# 体積の変わり方を測らない、休みの形の体積の小ささ（仮）。
# |休みの体積| <= この数 × (休みの表面積)^1.5 のとき測らない（灰）。平らな形・裏表が重なった形など。
# 球なら 体積 / 表面積^1.5 は約 0.094 なので、ふつうの閉じた形は、この数よりずっと大きい
VOLUME_MIN_RATIO = 1e-6
# とげの強さ：まわりの辺の長さの平均に対する、とび出しの割合（仮）
SPIKE_YELLOW = 0.5
SPIKE_RED = 1.0

# MARK: - 面の点検（安全な直し）

# 重なった点とみなす距離（Blender の「距離でマージ」の既定 0.0001 m と同じ）
MERGE_DISTANCE = mm(0.1)
# 面積 0 とみなす面積（平方メートル）（仮）
DEGENERATE_AREA = 1e-12
# 小さな離れた部品とみなす三角形の数（これ以下。いちばん大きい部品は除く）（仮）。
# 目のハイライトの板（三角形2つ）なども入るが、消すのはユーザーの確認のあとなのでよい
LOOSE_PART_MAX_FACES = 4

# MARK: - 関節と足（A4・M4）

# ひざ・ひじの逆曲がりとみなす角度（仮）
REVERSE_BEND_TOLERANCE_DEG = 2.0
# 太ももの位置だけからひざの軸を決めるとき、太ももとキャラの左右の向きがこの角度より近いと、軸が決まらないので測らない（灰）（仮）。
# 脚を真横に大きく開く・真横に上げる姿勢で、ふつうの曲がりを逆曲がり（赤）とまちがえないため
KNEE_AXIS_MIN_ANGLE_DEG = 25.0
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
# ↑ 速いダンスでは正しい動きでも超えることがある。本物の VMD で確かめてから決め直す（ユーザーと決めた）

# MARK: - 確かめの動き（A5）

# ポーズを付けたあと、骨の向きが決めた向きからこの角度（度）より離れていたら、モデルの拘束（コンストレイント）が
# ポーズを変えた（または D の骨が元の骨について来ない）とみなす。そのポーズの確かめは灰にする（仮）
POSE_FOLLOW_TOLERANCE_DEG = 1.0

# MARK: - 見える・見えない（A4）

# 見えるかどうかを調べる画像の大きさ（ピクセル、長い辺）（仮）
VISIBILITY_RESOLUTION = 512
# 奥行きの比べで、同じ面とみなす幅（仮）
VISIBILITY_DEPTH_EPSILON = mm(1.0)
# アプリのカメラの画角（KICKOFF の「画角70°」。縦か横かは、紅魔館のコードで確かめる）
APP_CAMERA_FOV_DEG = 70.0
# アプリの画面は縦長（横 ÷ 縦。iPhone の縦画面の目安）（仮）
APP_SCREEN_ASPECT = 9.0 / 19.5
# アプリでは、ユーザーがカメラの視点を動かせて、360 度どの向きからも見られる（ユーザーに確かめた）。
# そのため「アプリのカメラから見える」は、まわり全体から見るカメラ（orbit_views）のどれかから見えることとして調べる。
# 水平の向きの間隔（度）と、上下の角度（度。正は上から見下ろす）。真上と真下も足す（仮。数を増やすと重くなる）
APP_ORBIT_AZIMUTH_STEP_DEG = 30.0
APP_ORBIT_ELEVATIONS_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)
# 見え方を調べるカメラの、写す奥行きの範囲の手前（Blender の clip_start）。これより近い物は写さない（仮。アプリのカメラに合わせる）。
# Blender の初めの値は 0.1m だが、顔の近くに置くカメラでも切れないように短くする
CAMERA_NEAR = mm(10.0)
# 見え方を調べるカメラの、写す奥行きの範囲の奥（Blender の clip_end。初めの値と同じ 1000m）。これより遠い物は写さない（仮）
CAMERA_FAR = 1000.0
