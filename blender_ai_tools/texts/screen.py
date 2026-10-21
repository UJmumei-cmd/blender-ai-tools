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
    "screen.bones_sway": {"ja": "そのうち、揺れ物の骨：{count} 本", "en": "Of these, sway bones: {count}"},
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

    # MARK: - 確かめと書き出し（A6 版・E1＋E2・A5）
    "screen.needs_saved_file": {
        "ja": "先にファイルを保存してください（版・仕事・記録を、ファイルの横の作業フォルダに置くため）",
        "en": "Save the file first (versions, jobs and records are kept in the work folder next to the file)",
    },
    "screen.save_version": {"ja": "版を保存", "en": "Save version"},
    "screen.op_save_version": {"ja": "新しい版で保存する", "en": "Save as a new version"},
    "screen.save_version_tip": {
        "ja": "いまの中身を、新しい版（_v001、_v002 …）として保存し、形・骨・ウェイトの記録を残します。"
              "開いているファイルは上書きしません",
        "en": "Save the current contents as a new version (_v001, _v002 …) with a record of shapes, bones and "
              "weights. The open file is not overwritten",
    },
    "screen.save_version_failed": {"ja": "版を保存できませんでした：{error}", "en": "Could not save a version: {error}"},
    "screen.compare_versions": {"ja": "前の版と比べる", "en": "Compare"},
    "screen.op_compare_versions": {"ja": "前の版と比べる", "en": "Compare with the previous version"},
    "screen.compare_versions_tip": {
        "ja": "いまの中身を、いちばん新しい版の記録と比べて、変わった所を報告にします（何も変えません）",
        "en": "Compare the current contents with the record of the newest version and list the changes in the "
              "report (nothing is changed)",
    },
    "screen.compare_failed": {"ja": "前の版と比べられませんでした：{error}", "en": "Could not compare: {error}"},
    "screen.versions_title": {"ja": "保存した版（{count} 個）", "en": "Saved versions ({count})"},
    "screen.versions_none": {"ja": "まだ版を保存していません", "en": "No version has been saved yet"},
    "screen.versions_unsaved": {
        "ja": "ファイルを保存すると、版を保存できます",
        "en": "Save the file to be able to save versions",
    },
    "screen.versions_more": {"ja": "ほか {count} 個（古い版）", "en": "{count} more (older versions)"},
    "screen.export_and_check": {"ja": "書き出して確かめる", "en": "Export and check"},
    "screen.op_export_and_check": {"ja": "書き出して確かめる", "en": "Export and check"},
    "screen.export_and_check_tip": {
        "ja": "いちばん新しい保存した版を、紅魔館の convert_blend.py で作業フォルダに書き出し、読み戻して確かめます"
              "（E1＋E2。裏の Blender で動きます。アプリのフォルダには書きません）",
        "en": "Export the newest saved version with Koumakan's convert_blend.py into the work folder, read it back "
              "and check it (E1+E2; runs in a background Blender; nothing is written into the app folder)",
    },
    "screen.export_failed": {"ja": "書き出しを始められませんでした：{error}", "en": "Could not start the export: {error}"},
    "screen.export_input": {"ja": "書き出す版：{name}", "en": "Version to export: {name}"},
    "screen.export_needs_version": {
        "ja": "書き出す版がありません。先に「版を保存」を押してください（書き出すのは、保存した版です）",
        "en": "There is no version to export. Press “Save version” first (the saved version is exported)",
    },
    "screen.export_needs_koumakan": {
        "ja": "紅魔館の場所を、拡張機能の設定に書いてください（書き出しに要ります）",
        "en": "Set the Koumakan repository in the add-on preferences (needed to export)",
    },
    "screen.export_koumakan_invalid": {
        "ja": "紅魔館の場所が正しくありません：{error}",
        "en": "The Koumakan repository setting is not valid: {error}",
    },
    "screen.export_output_exists": {
        "ja": "出力フォルダがもうあります：{path}",
        "en": "The output folder already exists: {path}",
    },
    "screen.run_course": {"ja": "確かめの動き", "en": "Pose course"},
    "screen.op_run_course": {"ja": "確かめの動き", "en": "Run the pose course"},
    "screen.run_course_tip": {
        "ja": "いまの形の写しに、決まったポーズを付けて、ひざの逆曲がり・床への沈み・服からの体のはみ出しを確かめます"
              "（A5。裏の Blender で動きます。開いているファイルは保存しません）",
        "en": "Pose a copy of the current state and check reverse knee bends, sinking into the floor and the body "
              "showing through clothes (A5; runs in a background Blender; the open file is not saved)",
    },
    "screen.course_failed": {
        "ja": "確かめの動きを始められませんでした：{error}",
        "en": "Could not start the pose course: {error}",
    },
    "screen.course_copy_failed": {
        "ja": "いまの形の写しを保存できなかったので、仕事を取り消しました：{error}",
        "en": "The copy of the current state could not be saved, so the job was withdrawn: {error}",
    },
    "screen.course_note": {
        "ja": "いまの形の写しを、裏の Blender で確かめます（アプリのカメラから見えるかも調べるので、時間がかかります）",
        "en": "A copy of the current state is checked in a background Blender (it also checks what the app camera "
              "can see, so it takes a while)",
    },
    "screen.job_queued": {"ja": "仕事を順番待ちに入れました：{tool}", "en": "The job was queued: {tool}"},

    # MARK: - 裏の仕事（A10）
    "screen.section_checks": {"ja": "確かめと書き出し", "en": "Checks and export"},
    "screen.section_jobs": {"ja": "裏の仕事", "en": "Background jobs"},
    "screen.section_report": {"ja": "報告", "en": "Report"},
    "screen.tool_export_convert_and_check": {"ja": "書き出して確かめる", "en": "Export and check"},
    "screen.tool_motion_run_course": {"ja": "確かめの動き", "en": "Pose course"},
    "screen.jobs_none": {"ja": "仕事はありません", "en": "No jobs"},
    "screen.jobs_more": {"ja": "ほか {count} 件（古い仕事）", "en": "{count} more (older jobs)"},
    "screen.jobs_unreadable": {
        "ja": "読めない仕事のフォルダが {count} 個あります",
        "en": "{count} job folders cannot be read",
    },
    "screen.job_title": {"ja": "{tool}：{status}", "en": "{tool}: {status}"},
    "screen.job_created": {"ja": "出した時刻：{time}", "en": "Queued at: {time}"},
    "screen.job_cleaning": {"ja": "片付け中", "en": "Cleaning up"},
    "screen.job_step": {"ja": "{step} / {total}：{message}", "en": "{step} / {total}: {message}"},
    "screen.cancel_job": {"ja": "中止", "en": "Cancel"},
    "screen.op_cancel_job": {"ja": "仕事を中止する", "en": "Cancel the job"},
    "screen.cancel_job_tip": {
        "ja": "仕事を中止します（順番待ちならすぐ。動いていれば、裏の Blender に止まるように頼みます）",
        "en": "Cancel the job (at once when queued; when running, the background Blender is asked to stop)",
    },
    "screen.job_cancel_requested": {"ja": "仕事 {job_id} の中止を頼みました", "en": "Asked job {job_id} to stop"},
    "screen.remove_job": {"ja": "一覧から消す", "en": "Remove"},
    "screen.op_remove_job": {"ja": "仕事を一覧から消す", "en": "Remove the job from the list"},
    "screen.remove_job_tip": {
        "ja": "終わった仕事を一覧から消します（仕事のフォルダを消します。書き出したファイルは exports に残ります）",
        "en": "Remove a finished job from the list (its job folder is deleted; exported files stay in exports)",
    },
    "screen.job_removed": {"ja": "仕事 {job_id} を一覧から消しました", "en": "Removed job {job_id} from the list"},
    "screen.show_job_result": {"ja": "結果を見る", "en": "Show result"},
    "screen.op_show_job_result": {"ja": "仕事の結果を見る", "en": "Show the job result"},
    "screen.show_job_result_tip": {
        "ja": "終わった仕事の結果を、報告に出します",
        "en": "Show the result of the finished job in the report",
    },
    "screen.job_failed": {"ja": "仕事を扱えませんでした：{error}", "en": "Could not handle the job: {error}"},
    "screen.job_missing": {"ja": "その仕事はありません：{job_id}", "en": "There is no such job: {job_id}"},

    # MARK: - 報告（A11）
    "screen.report_none": {
        "ja": "まだ報告はありません（前の版と比べる・書き出して確かめる・確かめの動きのあとに出ます）",
        "en": "No report yet (it appears after comparing, exporting and checking, or the pose course)",
    },
    "screen.report_header": {"ja": "{kind}（{time}）", "en": "{kind} ({time})"},
    "screen.report_shown": {"ja": "報告：{headline}", "en": "Report: {headline}"},
    "screen.item_value": {"ja": "値：{value}", "en": "Value: {value}"},
    "screen.decision_current": {"ja": "答え：{decision}", "en": "Decision: {decision}"},
    "screen.prop_decision": {"ja": "答え", "en": "Decision"},
    "screen.op_decide": {"ja": "報告の項目に答える", "en": "Decide on a report item"},
    "screen.decide_tip": {
        "ja": "この項目を、直す・このままでよい・あとで のどれにするかを記録します（作業フォルダの decisions.json。"
              "「このままでよい」とした物は、次の版でも数えません）",
        "en": "Record whether to fix this item, leave it as is, or decide later (decisions.json in the work folder; "
              "items left as is are not counted in later versions either)",
    },
    "screen.op_clear_decision": {"ja": "答えを消す", "en": "Remove the decision"},
    "screen.clear_decision_tip": {
        "ja": "この項目の答えを消します（まだ決めていない、に戻します）",
        "en": "Remove the decision for this item (back to undecided)",
    },
    "screen.decided": {"ja": "「{title}」を「{decision}」にしました", "en": "Set “{title}” to “{decision}”"},
    "screen.decision_cleared": {"ja": "「{title}」の答えを消しました", "en": "Removed the decision for “{title}”"},
    "screen.decision_failed": {"ja": "答えを記録できませんでした：{error}", "en": "Could not record the decision: {error}"},
    "screen.decisions_unreadable": {
        "ja": "答えの記録（decisions.json）が読めません：{error}",
        "en": "The decision record (decisions.json) cannot be read: {error}",
    },
    "screen.no_report": {
        "ja": "報告がありません。先に、比べるか確かめてください",
        "en": "There is no report. Compare or run a check first",
    },
    "screen.item_missing": {"ja": "報告に、その項目がありません：{item}", "en": "The report has no such item: {item}"},
    "screen.write_report_html": {"ja": "HTML", "en": "HTML"},
    "screen.op_write_report_html": {"ja": "報告を HTML にする", "en": "Write the report as HTML"},
    "screen.write_report_html_tip": {
        "ja": "いまの報告を、作業フォルダの reports に HTML のファイルとして書きます",
        "en": "Write the current report as an HTML file into reports in the work folder",
    },
    "screen.open_report_html": {"ja": "開く", "en": "Open"},
    "screen.op_open_report_html": {"ja": "報告をブラウザで開く", "en": "Open the report in a browser"},
    "screen.open_report_html_tip": {
        "ja": "いまの報告を HTML に書き直して、ブラウザで開きます",
        "en": "Write the current report as HTML again and open it in a browser",
    },
    "screen.report_html_written": {"ja": "報告の HTML を書きました：{path}", "en": "Wrote the report HTML: {path}"},
    "screen.report_html_at": {"ja": "HTML：{path}", "en": "HTML: {path}"},
    "screen.report_html_failed": {
        "ja": "報告の HTML を書けませんでした：{error}",
        "en": "Could not write the report HTML: {error}",
    },
    "screen.record_baseline": {"ja": "基準にする", "en": "Set baseline"},
    "screen.op_record_baseline": {"ja": "赤の数を基準として記録する", "en": "Record the red count as the baseline"},
    "screen.record_baseline_tip": {
        "ja": "いまの赤の数（「このままでよい」は除く）を、次の版と比べる基準として記録します（増えたら赤）",
        "en": "Record the current red count (excluding items left as is) as the baseline for later versions "
              "(an increase is red)",
    },
    "screen.baseline_recorded": {
        "ja": "赤の数 {count} 件を、次の版と比べる基準として記録しました",
        "en": "Recorded {count} red as the baseline for later versions",
    },

    # MARK: - エラー
    "screen.unexpected_error": {
        "ja": "思いがけないエラーです（{error}）。くわしくは、コンソールを見てください",
        "en": "Unexpected error ({error}). See the console for details",
    },

    # MARK: - 拡張機能の設定
    "screen.pref_language": {"ja": "画面の言語", "en": "Language"},
    "screen.pref_language_tip": {
        "ja": "この道具の画面と報告の言語（初めは日本語。「Blender に合わせる」は、Blender が日本語なら日本語、ほかは英語）",
        "en": "Language of this tool's screens and reports (Japanese at first. “Follow Blender” gives Japanese when "
              "Blender is in Japanese, otherwise English)",
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
        "ja": "書き出しと裏の作業に使う Blender のファイル（Mac では Blender.app を選んでもよい）。"
              "空なら、いま動いている Blender を使います",
        "en": "Blender executable for export and background work (on Mac, Blender.app can be chosen). "
              "When empty, the running Blender is used",
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
