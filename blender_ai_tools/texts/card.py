"""モデルの登録カード（A1）の文。キー → {"ja": 日本語, "en": 英語}。数字などは {name} の形で入れる"""

TEXTS: dict[str, dict[str, str]] = {
    "card.bones_changed": {
        "ja": "骨（名前・並び・親子・位置・ねじれ・形を動かすかどうか）が変わりました",
        "en": "The bones changed (names, order, parents, positions, roll or deform flags)",
    },
    "card.shape_changed": {"ja": "{name} の形が変わりました", "en": "The shape of {name} changed"},
    "card.mesh_removed": {"ja": "{name} がなくなりました", "en": "{name} is gone"},
    "card.mesh_added": {"ja": "{name} が増えました", "en": "{name} was added"},
    "card.armature_changed": {
        "ja": "アーマチュアが変わりました（{before} → {after}）",
        "en": "The armature changed ({before} → {after})",
    },
    "card.no_card": {
        "ja": "登録カードがありません。先にモデルを登録してください",
        "en": "There is no registration card. Please register the model first",
    },
    "card.unreadable_card": {"ja": "登録カードが読めません：{error}", "en": "The registration card cannot be read: {error}"},
    "card.unreadable_model": {"ja": "いまのモデルを読めません：{error}", "en": "The current model cannot be read: {error}"},
    "card.legacy_card": {
        "ja": "古い形（版 {version}）の登録カードです。位置が世界の座標で入っているので、一度測り直してください",
        "en": "This registration card has an old format (version {version}) with world positions. "
              "Please measure the model once more",
    },
    "card.placement_moved": {
        "ja": "置き場所が変わりました（形はそのまま）",
        "en": "The model was moved (its shape is unchanged)",
    },
    "card.placement_resized": {
        "ja": "モデルの大きさか傾き（アーマチュアの置き方）が変わりました。背の高さと長さを測り直してください",
        "en": "The size or tilt of the model (the armature placement) changed. Please measure the height and "
              "lengths again",
    },
    "card.choices_not_carried": {
        "ja": "前のカード（アーマチュア {armature}）が、いまのモデルと同じモデルの物か確かめられない"
              "（アーマチュアの名前か骨が違う）ので、部品の種類・骨の答え・素材の出所と利用条件を写しませんでした",
        "en": "The previous card (armature {armature}) may belong to a different model (the armature name or the "
              "bones differ), so part kinds, bone answers and the source and terms of use were not copied",
    },
    "card.part_choice_dropped": {
        "ja": "部品 {name} がなくなったので、選んだ種類を写しませんでした",
        "en": "The part {name} is gone, so its chosen kind was not copied",
    },
    "card.material_choice_dropped": {
        "ja": "部品 {name} の材質 {material} がなくなったので、選んだ種類を写しませんでした",
        "en": "The material {material} of the part {name} is gone, so its chosen kind was not copied",
    },
    "card.answer_dropped": {
        "ja": "骨 {bone} の答え（{answer}）は、いまの辞書では使えないので外しました",
        "en": "The answer for the bone {bone} ({answer}) cannot be used with the current dictionary, so it was removed",
    },
    "card.semi_standard_missing": {
        "ja": "MMD 準標準の骨が足りないか、決まらない骨があります（その骨が要る確かめは灰になります）",
        "en": "Some MMD semi-standard bones are missing or undecided (checks that need them become gray)",
    },
    "card.unknown_bones": {
        "ja": "辞書に無い骨・1つに決まらない骨があります（ユーザーに聞きます）",
        "en": "Some bones are not in the dictionary or are undecided (the user is asked)",
    },
    "card.no_parts": {"ja": "部品（メッシュ）が1つもありません", "en": "There are no parts (meshes)"},
    "card.unconfirmed_parts": {
        "ja": "種類（服なら服の種類も）をまだ決めていない部品か材質があります",
        "en": "Some parts or materials have no confirmed kind yet (or no clothing type for clothes)",
    },
    "card.unmeasured_values": {"ja": "測れなかった値があります：{names}", "en": "Some values could not be measured: {names}"},
    "card.all_meshes_height": {
        "ja": "体の部品が決まっていないので、背の高さを全部のメッシュで測りました",
        "en": "No part is marked as body, so the height was measured with all meshes",
    },
    "card.source_missing": {
        "ja": "素材の出所と利用条件が書かれていません：{fields}",
        "en": "The source and terms of use are not written: {fields}",
    },
    "card.stale": {"ja": "測り直しが必要です：{reasons}", "en": "The card needs to be measured again: {reasons}"},
    "card.source_origin": {"ja": "素材の出所", "en": "Source"},
    "card.source_author": {"ja": "作者", "en": "Author"},
    "card.source_url": {"ja": "URL", "en": "URL"},
    "card.source_terms": {"ja": "利用条件", "en": "Terms of use"},
    "card.list_separator": {"ja": "、", "en": ", "},
    "card.sentence_separator": {"ja": "。", "en": ". "},
}
