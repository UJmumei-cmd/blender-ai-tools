"""文の形（日本語・英語）。キー → {"ja": 日本語, "en": 英語}。数字などは {name} の形で入れる

A5 確かめの動き（motion/）の、ポーズの名前・確かめの名前・理由・報告の文。
"""

TEXTS: dict[str, dict[str, str]] = {
    # MARK: - ポーズの名前（motion.poses.POSES の text_key）
    # 関節の動く範囲
    "motion.pose_head_down": {"ja": "頭を下げる", "en": "Head down"},
    "motion.pose_head_up": {"ja": "上を向く", "en": "Looking up"},
    "motion.pose_head_turn_left": {"ja": "左を向く", "en": "Head turned left"},
    "motion.pose_head_turn_right": {"ja": "右を向く", "en": "Head turned right"},
    "motion.pose_head_tilt_left": {"ja": "首を左に傾ける", "en": "Head tilted left"},
    "motion.pose_head_tilt_right": {"ja": "首を右に傾ける", "en": "Head tilted right"},
    "motion.pose_torso_forward": {"ja": "体を前に曲げる", "en": "Bending forward"},
    "motion.pose_torso_back": {"ja": "体を後ろに反らす", "en": "Bending backward"},
    "motion.pose_torso_side_left": {"ja": "体を左に傾ける", "en": "Leaning left"},
    "motion.pose_torso_side_right": {"ja": "体を右に傾ける", "en": "Leaning right"},
    "motion.pose_torso_twist_left": {"ja": "体を左にひねる", "en": "Twisting left"},
    "motion.pose_torso_twist_right": {"ja": "体を右にひねる", "en": "Twisting right"},
    "motion.pose_shoulders_up": {"ja": "肩をすくめる", "en": "Shrugging"},
    "motion.pose_arms_down": {"ja": "腕を体に付ける", "en": "Arms at the sides"},
    "motion.pose_arms_back": {"ja": "腕を後ろに引く", "en": "Arms pulled back"},
    "motion.pose_elbows_bend": {"ja": "ひじを深く曲げる", "en": "Elbows fully bent"},
    "motion.pose_forearms_twist": {"ja": "前腕をひねる", "en": "Forearms twisted"},
    "motion.pose_fists": {"ja": "こぶしをにぎる", "en": "Fists"},
    "motion.pose_leg_front_left": {"ja": "左脚を前に上げる", "en": "Left leg raised forward"},
    "motion.pose_leg_front_right": {"ja": "右脚を前に上げる", "en": "Right leg raised forward"},
    "motion.pose_leg_side_left": {"ja": "左脚を横に開く", "en": "Left leg out to the side"},
    "motion.pose_leg_side_right": {"ja": "右脚を横に開く", "en": "Right leg out to the side"},
    "motion.pose_leg_back_left": {"ja": "左脚を後ろに引く", "en": "Left leg back"},
    "motion.pose_leg_back_right": {"ja": "右脚を後ろに引く", "en": "Right leg back"},
    "motion.pose_knee_bend_left": {"ja": "左ひざを深く曲げる", "en": "Left knee fully bent"},
    "motion.pose_knee_bend_right": {"ja": "右ひざを深く曲げる", "en": "Right knee fully bent"},
    "motion.pose_tiptoe": {"ja": "つま先立ち", "en": "On tiptoe"},
    # ふだんの動き
    "motion.pose_arms_up": {"ja": "両手を上げる", "en": "Arms up"},
    "motion.pose_arms_crossed": {"ja": "腕を組む", "en": "Arms crossed"},
    "motion.pose_clap": {"ja": "手をたたく", "en": "Clapping"},
    "motion.pose_hands_on_cheeks": {"ja": "ほおに手を当てる", "en": "Hands on the cheeks"},
    "motion.pose_hand_on_chin": {"ja": "あごに手を当てる", "en": "Hand on the chin"},
    "motion.pose_bow": {"ja": "おじぎ", "en": "Bowing"},
    "motion.pose_look_back": {"ja": "振り返る", "en": "Looking back"},
    "motion.pose_walk": {"ja": "歩く（一歩の途中）", "en": "Walking (mid-stride)"},
    "motion.pose_run": {"ja": "走る（一歩の途中）", "en": "Running (mid-stride)"},
    "motion.pose_squat": {"ja": "しゃがむ", "en": "Squatting"},
    "motion.pose_chair_sit": {"ja": "いすの端に座る", "en": "Sitting on the edge of a chair"},
    "motion.pose_seiza": {"ja": "正座", "en": "Seiza (kneeling)"},
    "motion.pose_hands_on_knees": {"ja": "ひざに手をつく", "en": "Hands on the knees"},
    "motion.pose_wave": {"ja": "手を振る", "en": "Waving"},
    "motion.pose_sword_stance": {"ja": "刀を構える", "en": "Sword stance"},
    # 横になる
    "motion.pose_lying_back": {"ja": "あおむけに寝る", "en": "Lying on the back"},
    "motion.pose_lying_side": {"ja": "横向きに寝る", "en": "Lying on the side"},
    # MARK: - ポーズの組
    "motion.category_rom": {"ja": "関節の動く範囲", "en": "Range of motion"},
    "motion.category_daily": {"ja": "ふだんの動き", "en": "Everyday poses"},
    "motion.category_lying": {"ja": "横になる", "en": "Lying down"},
    # MARK: - アプリの動き
    "motion.app_motions_all": {"ja": "アプリの動き（全部）", "en": "App motions (all)"},
    "motion.app_motion": {"ja": "アプリの動き「{name}」", "en": "App motion \"{name}\""},
    # MARK: - 確かめの名前
    "motion.check_knee_reverse_bend": {"ja": "ひざの逆曲がり", "en": "Reverse knee bend"},
    "motion.check_floor_sink": {"ja": "床への沈み", "en": "Sinking into the floor"},
    "motion.check_cloth_penetration": {"ja": "服からの体のはみ出し", "en": "Body poking through the clothes"},
    "motion.check_app_motion": {"ja": "アプリの動き", "en": "App motion"},
    # MARK: - 信号
    "motion.signal_green": {"ja": "緑", "en": "Green"},
    "motion.signal_yellow": {"ja": "黄", "en": "Yellow"},
    "motion.signal_gray": {"ja": "灰", "en": "Gray"},
    "motion.signal_red": {"ja": "赤", "en": "Red"},
    # MARK: - 理由（灰など）
    "motion.reason_missing_bones": {
        "ja": "ポーズに要る骨がありません（{bones}）。決めたとおりのポーズで測れていません",
        "en": "Bones needed for this pose are missing ({bones}). The intended pose could not be measured",
    },
    "motion.reason_overridden": {
        "ja": "モデルの拘束で、骨が決めた向きになりませんでした（{bones}）。決めたとおりのポーズで測れていません",
        "en": "The model's constraints kept these bones from the intended pose ({bones}). "
              "The intended pose could not be measured",
    },
    "motion.reason_app_motions": {
        "ja": "アプリの動き（motions.json）は、MotionLibrary.swift が無いので、まだ計算できません",
        "en": "The app motions (motions.json) cannot be computed yet because MotionLibrary.swift is missing",
    },
    "motion.reason_no_body": {
        "ja": "体の部品がありません（登録カードで、体の部品を選んでください）",
        "en": "There is no body part (choose the body parts in the model card)",
    },
    "motion.reason_no_cloth": {
        "ja": "服の部品がありません（服が無いか、登録カードで服を選んでいません）",
        "en": "There is no clothing part (the model has no clothes, or none is chosen in the model card)",
    },
    "motion.reason_unreadable_meshes": {
        "ja": "形を読めないメッシュがあります（{names}）",
        "en": "Some meshes could not be read ({names})",
    },
    "motion.reason_no_leg_bones": {
        "ja": "ひざを測る骨がありません（{bones}）",
        "en": "Bones for measuring the knees are missing ({bones})",
    },
    "motion.reason_knee_axis": {
        "ja": "休みの姿勢で、ひざの軸が決まりません（{sides}）",
        "en": "The knee axis cannot be found in the rest pose ({sides})",
    },
    "motion.reason_no_closed_body": {
        "ja": "体の部品に、閉じた形がありません。服の点が体の中か外か決められません",
        "en": "No body part is a closed shape, so it cannot be told whether clothing points are inside the body",
    },
    "motion.reason_open_body": {
        "ja": "閉じていない体の部品が {count} 個あります。その近くの服の点は、体の中か外か決められません",
        "en": "{count} body part(s) are not closed shapes. Clothing points near them cannot be judged inside or outside",
    },
    "motion.reason_pose_failed": {
        "ja": "ポーズの形を計算できませんでした：{error}",
        "en": "The posed shape could not be computed: {error}",
    },
    "motion.reason_no_events": {"ja": "何も測っていません", "en": "Nothing was measured"},
    "motion.summary_reason": {
        "ja": "ポーズ {poses} 個：赤の出来事が {red} 件、灰が {gray} 件、黄が {yellow} 件あります",
        "en": "{poses} pose(s): {red} red, {gray} gray and {yellow} yellow event(s)",
    },
    "motion.summary_all_green": {
        "ja": "ポーズ {poses} 個：どの出来事も緑です",
        "en": "{poses} pose(s): every event is green",
    },
    # MARK: - 報告の文
    "motion.report_header": {
        "ja": "確かめの動き：ポーズ {poses} 個、出来事 {events} 件（赤 {red}・灰 {gray}・黄 {yellow}・緑 {green}）",
        "en": "Pose course: {poses} pose(s), {events} event(s) (red {red}, gray {gray}, yellow {yellow}, green {green})",
    },
    "motion.report_event": {"ja": "{signal}　{pose}：{check}　{value}", "en": "{signal}  {pose}: {check}  {value}"},
    "motion.report_event_reason": {
        "ja": "{signal}　{pose}：{check}　{value}（{reason}）",
        "en": "{signal}  {pose}: {check}  {value} ({reason})",
    },
    "motion.report_all_green": {"ja": "赤・灰・黄の出来事はありません", "en": "There are no red, gray or yellow events"},
    "motion.value_mm": {"ja": "{value} mm", "en": "{value} mm"},
    "motion.value_deg": {"ja": "{value} 度", "en": "{value}°"},
    "motion.value_count": {"ja": "{value} 個", "en": "{value}"},
    "motion.value_other": {"ja": "{value} {unit}", "en": "{value} {unit}"},
    "motion.value_none": {"ja": "値なし", "en": "no value"},
    "motion.list_separator": {"ja": "、", "en": ", "},
}
