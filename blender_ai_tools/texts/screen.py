"""画面の文の形（日本語・英語）。キー → {"ja": 日本語, "en": 英語}。数字などは {name} の形で入れる。

ui/ のパネル・ボタン・説明（ツールチップ）・報告の文と、拡張機能の設定の文をここに置く。
数字の文字は、呼ぶ側（プログラム）が作って入れる。
部品の種類（screen.kind_*）と服の種類（screen.cloth_*）は、model.card の PART_KINDS・CLOTH_TYPE_IDS と
同じ名前をそろえる（tests/unit/test_screen_texts.py が確かめる）。
"""

TEXTS: dict[str, dict[str, str]] = {
    # MARK: - タブとパネル
    "screen.tab": {"ja": "AI の道具", "en": "AI Tools"},
    "screen.panel_title": {"ja": "AI の道具", "en": "AI Tools"},
    "screen.section_model": {"ja": "モデル", "en": "Model"},
    "screen.section_bones": {"ja": "骨", "en": "Bones"},
    "screen.section_next": {"ja": "書き出し・確かめの動き・版", "en": "Export, test motions, versions"},
    "screen.stage_note": {
        "ja": "段階1：ボタンだけで動きます（AI はつないでいません）",
        "en": "Stage 1: everything works with buttons (no AI is connected)",
    },
    "screen.model_found": {
        "ja": "モデル：{armature}（部品 {meshes} 個）",
        "en": "Model: {armature} ({meshes} parts)",
    },
    "screen.model_not_found": {"ja": "モデルが見つかりません：{error}", "en": "No model was found: {error}"},

    # MARK: - 信号
    "screen.signal_green": {"ja": "緑", "en": "Green"},
    "screen.signal_yellow": {"ja": "黄", "en": "Yellow"},
    "screen.signal_gray": {"ja": "灰", "en": "Gray"},
    "screen.signal_red": {"ja": "赤", "en": "Red"},
    "screen.signal_row": {"ja": "【{signal}】{text}", "en": "[{signal}] {text}"},
    "screen.not_measured": {"ja": "測れていません", "en": "not measured"},
    "screen.list_separator": {"ja": "、", "en": ", "},

    # MARK: - モデル（A1 登録カード）
    "screen.register_card": {"ja": "登録する", "en": "Register"},
    "screen.op_register_card": {"ja": "モデルを登録する", "en": "Register the model"},
    "screen.register_card_tip": {
        "ja": "モデルを測って、登録カードをこのファイルの中に置きます。前に決めた種類と答えは写します"
              "（.blend のファイルは保存しません）",
        "en": "Measure the model and put its registration card inside this file. Earlier choices and answers are "
              "kept (the .blend file is not saved)",
    },
    "screen.check_card": {"ja": "確かめる", "en": "Check"},
    "screen.op_check_card": {"ja": "登録カードを確かめる", "en": "Check the registration card"},
    "screen.check_card_tip": {
        "ja": "登録カードが、いまのモデルにまだ合うかを確かめます（何も変えません）",
        "en": "Check whether the registration card still matches the current model (nothing is changed)",
    },
    "screen.no_card": {
        "ja": "まだ登録していません。「登録する」を押してください",
        "en": "Not registered yet. Press “Register”",
    },
    "screen.card_unreadable": {"ja": "登録カードが読めません：{error}", "en": "The registration card cannot be read: {error}"},
    "screen.card_legacy": {
        "ja": "古い形の登録カードです。「登録する」で、もう一度登録してください",
        "en": "This registration card has an old format. Press “Register” to register again",
    },
    "screen.summary_title": {"ja": "まとめ", "en": "Summary"},
    "screen.height": {"ja": "背の高さ：{value} mm", "en": "Height: {value} mm"},
    "screen.height_unmeasured": {
        "ja": "背の高さ：測れませんでした（{reason}）",
        "en": "Height: could not be measured ({reason})",
    },
    "screen.compliance_ok": {"ja": "MMD 準標準の骨：そろっています", "en": "MMD semi-standard bones: complete"},
    "screen.compliance_missing": {
        "ja": "MMD 準標準の骨：足りない骨 {missing} 本、決まらない骨 {conflicts} 本",
        "en": "MMD semi-standard bones: {missing} missing, {conflicts} undecided",
    },
    "screen.summary_unknown_bones": {
        "ja": "辞書に無い・決まらない骨：{count} 本",
        "en": "Bones not in the dictionary or undecided: {count}",
    },
    "screen.summary_unconfirmed_parts": {
        "ja": "種類を決めていない部品・材質：{count} 個",
        "en": "Parts or materials without a chosen kind: {count}",
    },
    "screen.summary_body_measurements": {
        "ja": "測れなかった値：{count} 個",
        "en": "Values that could not be measured: {count}",
    },
    "screen.summary_source_terms": {
        "ja": "書いていない素材の出所・利用条件：{count} 項目",
        "en": "Unwritten source and terms of use: {count}",
    },
    "screen.summary_card_current": {
        "ja": "測り直しが要る理由：{count} 個",
        "en": "Reasons to measure again: {count}",
    },
    "screen.summary_generic": {"ja": "{name}：{count}", "en": "{name}: {count}"},
    "screen.check_title": {"ja": "確かめた結果", "en": "Check result"},
    "screen.register_title": {"ja": "登録したときの知らせ", "en": "Notes from registering"},
    "screen.check_ok": {
        "ja": "登録カードは、いまのモデルに合っています",
        "en": "The registration card matches the current model",
    },
    "screen.check_stale": {
        "ja": "測り直しが必要です（理由 {count} 個）。「登録する」で、もう一度測ってください",
        "en": "The model needs to be measured again ({count} reasons). Press “Register” to measure again",
    },
    "screen.check_failed": {"ja": "確かめられませんでした：{error}", "en": "Could not check: {error}"},
    "screen.card_registered": {
        "ja": "登録しました：部品 {parts} 個、骨 {bones} 本",
        "en": "Registered: {parts} parts, {bones} bones",
    },
    "screen.register_failed": {"ja": "登録できませんでした：{error}", "en": "Could not register: {error}"},
    "screen.previous_card_unreadable": {
        "ja": "前の登録カードが読めないので、前に決めた物を写せませんでした。前のカードは {backup} に残しました",
        "en": "The previous registration card could not be read, so earlier choices were not copied. "
              "It was kept as {backup}",
    },

    # MARK: - 部品と材質の種類
    "screen.parts_title": {"ja": "部品（{count} 個）", "en": "Parts ({count})"},
    "screen.more_rows": {
        "ja": "ほか {count} 件（上から決めると、ここに出てきます）",
        "en": "{count} more (they appear here as you decide the ones above)",
    },
    "screen.kind_body": {"ja": "体", "en": "Body"},
    "screen.kind_cloth": {"ja": "服", "en": "Clothes"},
    "screen.kind_hair": {"ja": "髪", "en": "Hair"},
    "screen.kind_accessory": {"ja": "アクセサリー", "en": "Accessory"},
    "screen.kind_wing": {"ja": "羽・翼", "en": "Wings"},
    "screen.kind_other": {"ja": "その他", "en": "Other"},
    "screen.kind_unknown": {"ja": "分からない", "en": "Unknown"},
    "screen.kind_guess": {"ja": "{kind}（推測）", "en": "{kind} (guess)"},
    "screen.cloth_top": {"ja": "上着", "en": "Top"},
    "screen.cloth_shirt": {"ja": "シャツ・ブラウス", "en": "Shirt / blouse"},
    "screen.cloth_skirt": {"ja": "スカート", "en": "Skirt"},
    "screen.cloth_pants": {"ja": "ズボン", "en": "Trousers"},
    "screen.cloth_dress": {"ja": "ワンピース", "en": "Dress"},
    "screen.cloth_socks": {"ja": "靴下", "en": "Socks"},
    "screen.cloth_shoes": {"ja": "靴", "en": "Shoes"},
    "screen.cloth_gloves": {"ja": "手袋", "en": "Gloves"},
    "screen.cloth_hat": {"ja": "帽子", "en": "Hat"},
    "screen.cloth_other": {"ja": "その他", "en": "Other"},
    "screen.cloth_type_none": {"ja": "服の種類を選ぶ", "en": "Choose the clothing type"},
    "screen.prop_kind": {"ja": "種類", "en": "Kind"},
    "screen.prop_cloth_type": {"ja": "服の種類", "en": "Clothing type"},
    "screen.op_set_part_kind": {"ja": "部品の種類を選ぶ", "en": "Choose the part kind"},
    "screen.set_part_kind_tip": {
        "ja": "部品（材質）の種類を一覧から選びます。選んだ種類は「ユーザーが決めた種類」になります",
        "en": "Choose the kind of the part (material) from the list. It becomes a kind chosen by the user",
    },
    "screen.op_set_cloth_type": {"ja": "服の種類を選ぶ", "en": "Choose the clothing type"},
    "screen.set_cloth_type_tip": {
        "ja": "服の種類を一覧から選びます（種類が「服」の部品と材質だけ）",
        "en": "Choose the clothing type from the list (only for parts and materials whose kind is clothes)",
    },
    "screen.kind_set": {"ja": "{name} の種類を「{kind}」にしました", "en": "Set the kind of {name} to “{kind}”"},
    "screen.material_kind_set": {
        "ja": "{name} の材質 {material} の種類を「{kind}」にしました",
        "en": "Set the kind of the material {material} of {name} to “{kind}”",
    },
    "screen.cloth_type_set": {
        "ja": "{name} の服の種類を「{cloth}」にしました",
        "en": "Set the clothing type of {name} to “{cloth}”",
    },
    "screen.material_cloth_type_set": {
        "ja": "{name} の材質 {material} の服の種類を「{cloth}」にしました",
        "en": "Set the clothing type of the material {material} of {name} to “{cloth}”",
    },
    "screen.cloth_type_needs_cloth": {
        "ja": "服の種類は、種類が「服」の部品と材質だけに選べます（{name}）",
        "en": "A clothing type can only be chosen for parts and materials whose kind is clothes ({name})",
    },
    "screen.no_card_error": {
        "ja": "登録カードがありません。先に「登録する」を押してください",
        "en": "There is no registration card. Press “Register” first",
    },
    "screen.legacy_card_error": {
        "ja": "古い形の登録カードは変えられません。先に「登録する」で、もう一度登録してください",
        "en": "An old-format registration card cannot be changed. Press “Register” to register again first",
    },
    "screen.card_change_failed": {
        "ja": "登録カードを変えられませんでした：{error}",
        "en": "Could not change the registration card: {error}",
    },

    # MARK: - 骨（A3 骨の名前の対応）
    "screen.check_bones": {"ja": "骨の名前を調べる", "en": "Check bone names"},
    "screen.op_check_bones": {"ja": "骨の名前を調べる", "en": "Check bone names"},
    "screen.check_bones_tip": {
        "ja": "骨の名前を、MMD 準標準の名前に対応させます（道具の目録の「骨の名前の対応」を使います。何も変えません）",
        "en": "Match the bone names to the MMD semi-standard names (uses the “bone name mapping” tool from the "
              "tool catalog; nothing is changed)",
    },
    "screen.bones_checked": {
        "ja": "骨 {total} 本：対応 {mapped} 本、揺れ物など {extras} 本、分からない {unknown} 本、取り合い {conflicts} 個、"
              "足りない準標準の骨 {missing} 本",
        "en": "{total} bones: {mapped} matched, {extras} sway or other, {unknown} unknown, {conflicts} conflicts, "
              "{missing} missing semi-standard bones",
    },
    "screen.bones_check_failed": {
        "ja": "骨の名前を調べられませんでした：{error}",
        "en": "Could not check the bone names: {error}",
    },
    "screen.bones_check_title": {"ja": "調べた結果（{armature}）", "en": "Result ({armature})"},
    "screen.bones_total": {"ja": "骨：{count} 本", "en": "Bones: {count}"},
    "screen.bones_mapped": {"ja": "準標準の名前に対応：{count} 本", "en": "Matched to semi-standard names: {count}"},
    "screen.bones_extras": {"ja": "揺れ物・飾りなど：{count} 本", "en": "Sway, decoration and other bones: {count}"},
    "screen.bones_unknown": {"ja": "分からない骨：{count} 本", "en": "Unknown bones: {count}"},
    "screen.bones_conflicts": {"ja": "取り合い：{count} 個", "en": "Conflicts: {count}"},
    "screen.bones_missing": {"ja": "足りない準標準の骨：{count} 本", "en": "Missing semi-standard bones: {count}"},
    "screen.bones_answered": {
        "ja": "そのうち、登録カードで答えた骨：{count} 本",
        "en": "Of these, answered on the registration card: {count}",
    },
    "screen.questions_title": {"ja": "聞きたい骨（{count} 本）", "en": "Bones to answer ({count})"},
    "screen.no_questions": {"ja": "聞きたい骨はありません", "en": "No bones need an answer"},
    "screen.answers_need_card": {
        "ja": "答えを覚えるには、先にモデルを登録してください",
        "en": "Register the model first to store answers",
    },
    "screen.answer_choose": {"ja": "選ぶ…", "en": "Choose…"},
    "screen.answer_choose_tip": {
        "ja": "共通の名前（MMD 準標準）の一覧から、この骨に合う名前を選びます",
        "en": "Choose the matching common name (MMD semi-standard) for this bone from the list",
    },
    "screen.answer_extra": {"ja": "揺れ物・人の形ではない", "en": "Sway / not humanoid"},
    "screen.answer_extra_tip": {
        "ja": "この骨は、人の形の骨ではありません（髪・スカートなどの揺れ物や、飾りの骨）",
        "en": "This bone is not a humanoid bone (sway bones such as hair or skirt, or decoration bones)",
    },
    "screen.answer_clear_tip": {"ja": "この骨の答えを消します", "en": "Remove the answer for this bone"},
    "screen.answered": {"ja": "答え：{answer}", "en": "Answer: {answer}"},
    "screen.candidates": {"ja": "候補：", "en": "Candidates:"},
    "screen.candidates_list": {"ja": "候補：{names}", "en": "Candidates: {names}"},
    "screen.conflict_with": {"ja": "取り合い：{canonical}", "en": "Conflict: {canonical}"},
    "screen.prop_answer": {"ja": "共通の名前", "en": "Common name"},
    "screen.op_answer_bone": {"ja": "骨の答えを選ぶ", "en": "Answer a bone"},
    "screen.op_clear_bone_answer": {"ja": "骨の答えを消す", "en": "Remove a bone answer"},
    "screen.answer_set": {
        "ja": "骨 {bone} の答えを「{answer}」にしました",
        "en": "Set the answer for the bone {bone} to “{answer}”",
    },
    "screen.answer_cleared": {"ja": "骨 {bone} の答えを消しました", "en": "Removed the answer for the bone {bone}"},
    "screen.answer_failed": {"ja": "答えを入れられませんでした：{error}", "en": "Could not store the answer: {error}"},
    "screen.missing_title": {
        "ja": "足りない準標準の骨（{count} 本。その骨が要る確かめは灰になります）",
        "en": "Missing semi-standard bones ({count}; checks that need them become gray)",
    },

    # MARK: - 次の段階（A10 から）
    "screen.next_export": {"ja": "書き出す", "en": "Export"},
    "screen.next_export_tip": {
        "ja": "紅魔館の convert_blend.py で、アプリ用に書き出します（E1。裏で動かす係 A10 ができたら使えます）",
        "en": "Export for the app with Koumakan's convert_blend.py (E1; available once the background worker A10 "
              "is ready)",
    },
    "screen.next_motion_check": {"ja": "確かめの動き", "en": "Test motions"},
    "screen.next_motion_check_tip": {
        "ja": "決まったポーズとアプリの動きで、壊れないかを確かめます（A5。裏で動かす係 A10 ができたら使えます）",
        "en": "Check the model with fixed poses and the app's motions (A5; available once the background worker "
              "A10 is ready)",
    },
    "screen.next_versions": {"ja": "版を保存する", "en": "Save a version"},
    "screen.next_versions_tip": {
        "ja": "確かめたあとで、新しい版（_v001 など）として保存します（A6。裏で動かす係 A10 ができたら使えます）",
        "en": "Save a checked result as a new version (_v001 and so on) (A6; available once the background "
              "worker A10 is ready)",
    },
    "screen.next_note": {
        "ja": "これらは、裏で動かす係（A10）ができたら、別の Blender で動きます",
        "en": "These will run in a separate Blender through the background worker (A10), coming next",
    },
    "screen.op_coming_soon": {"ja": "まだ使えない道具", "en": "Not available yet"},
    "screen.coming_soon": {
        "ja": "まだ使えません。裏で動かす係（A10）ができたら使えます",
        "en": "Not available yet. It becomes available with the background worker (A10)",
    },

    # MARK: - エラー
    "screen.unexpected_error": {
        "ja": "思いがけないエラーです（{error}）。くわしくは、コンソールを見てください",
        "en": "Unexpected error ({error}). See the console for details",
    },

    # MARK: - 拡張機能の設定
    "screen.pref_language": {"ja": "画面の言語", "en": "Language"},
    "screen.pref_language_tip": {
        "ja": "この道具の画面と報告の言語（Blender に合わせる：Blender が日本語なら日本語、ほかは英語）",
        "en": "Language of this tool's screens and reports (Follow Blender: Japanese when Blender is in Japanese, "
              "otherwise English)",
    },
    "screen.pref_language_auto": {"ja": "Blender に合わせる", "en": "Follow Blender"},
    "screen.pref_language_ja": {"ja": "日本語", "en": "日本語 (Japanese)"},
    "screen.pref_language_en": {"ja": "English（英語）", "en": "English"},
    "screen.pref_koumakan_repo": {"ja": "紅魔館の場所", "en": "Koumakan repository"},
    "screen.pref_koumakan_repo_tip": {
        "ja": "紅魔館のリポジトリのフォルダ（tools/model-converter/convert_blend.py があるところ）",
        "en": "Folder of the Koumakan repository (the one with tools/model-converter/convert_blend.py)",
    },
    "screen.pref_koumakan_empty": {
        "ja": "まだ書いていません（アプリへの書き出しに要ります）",
        "en": "Not set yet (needed to export for the app)",
    },
    "screen.pref_koumakan_found": {"ja": "convert_blend.py が見つかりました", "en": "convert_blend.py was found"},
    "screen.pref_koumakan_missing": {"ja": "見つかりません：{error}", "en": "Not found: {error}"},
    "screen.pref_blender_path": {"ja": "書き出しに使う Blender", "en": "Blender used for export"},
    "screen.pref_blender_path_tip": {
        "ja": "書き出しと裏の作業に使う Blender のファイル。空なら、いま動いている Blender を使います",
        "en": "Blender executable for export and background work. When empty, the running Blender is used",
    },
    "screen.pref_blender_default": {
        "ja": "空なら、いま動いている Blender を使います：{path}",
        "en": "When empty, the running Blender is used: {path}",
    },
    "screen.pref_work_dir": {"ja": "作業フォルダ", "en": "Work folder"},
    "screen.pref_work_dir_tip": {
        "ja": "写し・書き出し・記録を置くフォルダ。空なら、.blend の横に作ります",
        "en": "Folder for copies, exports and records. When empty, one is made next to the .blend",
    },
    "screen.pref_work_dir_default": {
        "ja": "空なら、.blend の横の「（ファイルの名前）{suffix}」を使います",
        "en": "When empty, “(file name){suffix}” next to the .blend is used",
    },
    "screen.pref_work_dir_now": {"ja": "いまのファイルでは：{path}", "en": "For the current file: {path}"},
    "screen.pref_work_dir_unsaved": {
        "ja": "いまのファイルは、まだ保存していません（保存すると決まります）",
        "en": "The current file is not saved yet (it is decided when you save)",
    },
}
