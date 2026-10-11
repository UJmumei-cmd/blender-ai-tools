"""A6 版の管理と変更点の一覧の文の形（日本語・英語）。キー → {"ja": 日本語, "en": 英語}。数字などは {name} の形で入れる。

数字の形（小数の桁など）は、ここで {distance_mm:.3f} のように決める。数字そのものは、呼ぶ側（プログラム）が測って入れる。
名前の並び（{names} など）は、versions.diff.change_lines が、言語ごとの区切り（versions.list_separator）でつないでから入れる。
"""

TEXTS: dict[str, dict[str, str]] = {
    # MARK: - 一覧の形
    "versions.list_separator": {"ja": "、", "en": ", "},
    "versions.more_names": {"ja": "{names} ほか {count} 個", "en": "{names} and {count} more"},
    "versions.none_name": {"ja": "（なし）", "en": "(none)"},
    "versions.outside_mark": {"ja": "【変えてよい所の外】{line}", "en": "[Outside the allowed changes] {line}"},
    "versions.no_changes": {"ja": "前の版から変わった所はありません", "en": "Nothing changed since the previous version"},

    # MARK: - オブジェクト
    "versions.object_added": {
        "ja": "{object}：オブジェクトが増えました（種類 {type}）",
        "en": "{object}: the object was added (type {type})",
    },
    "versions.object_removed": {
        "ja": "{object}：オブジェクトが無くなりました（種類 {type}）",
        "en": "{object}: the object was removed (type {type})",
    },
    "versions.object_type_changed": {
        "ja": "{object}：オブジェクトの種類が変わりました（{old} → {new}）",
        "en": "{object}: the object type changed ({old} → {new})",
    },
    "versions.object_parent_changed": {
        "ja": "{object}：親が変わりました（{old} → {new}）",
        "en": "{object}: the parent changed ({old} → {new})",
    },
    "versions.object_moved": {
        "ja": "{object}：置き方（位置・回転・大きさ）が変わりました（位置のずれ {distance_mm:.3f} mm、"
              "回転・大きさの成分の差 {linear:.6g}）",
        "en": "{object}: the placement (location, rotation, scale) changed (location shift {distance_mm:.3f} mm, "
              "rotation/scale component difference {linear:.6g})",
    },

    # MARK: - 頂点と形のつながり
    "versions.vertices_moved": {
        "ja": "{object}：頂点が {count} 個動きました（いちばん大きい動き {distance_mm:.3f} mm）",
        "en": "{object}: {count} vertices moved (largest move {distance_mm:.3f} mm)",
    },
    "versions.topology_changed": {
        "ja": "{object}：形のつながりが変わりました（頂点 {old_vertices} → {new_vertices} 個、面 {old_faces} → {new_faces} 個）",
        "en": "{object}: the mesh topology changed (vertices {old_vertices} → {new_vertices}, "
              "faces {old_faces} → {new_faces})",
    },

    # MARK: - シェイプキー
    "versions.shape_keys_added": {
        "ja": "{object}：シェイプキーが {count} 個増えました（{names}）",
        "en": "{object}: {count} shape keys were added ({names})",
    },
    "versions.shape_keys_removed": {
        "ja": "{object}：シェイプキーが {count} 個無くなりました（{names}）",
        "en": "{object}: {count} shape keys were removed ({names})",
    },
    "versions.shape_key_changed": {
        "ja": "{object}：シェイプキー「{name}」の形が変わりました（頂点 {count} 個、いちばん大きい違い {distance_mm:.3f} mm）",
        "en": "{object}: the shape of shape key “{name}” changed ({count} vertices, "
              "largest difference {distance_mm:.3f} mm)",
    },
    "versions.shape_key_order_changed": {
        "ja": "{object}：シェイプキーの並びが変わりました（{names}）",
        "en": "{object}: the order of the shape keys changed ({names})",
    },

    # MARK: - ウェイト
    "versions.groups_added": {
        "ja": "{object}：頂点グループが {count} 個増えました（{names}）",
        "en": "{object}: {count} vertex groups were added ({names})",
    },
    "versions.groups_removed": {
        "ja": "{object}：頂点グループが {count} 個無くなりました（{names}）",
        "en": "{object}: {count} vertex groups were removed ({names})",
    },
    "versions.weights_changed": {
        "ja": "{object}：ウェイトが変わった頂点が {count} 個あります（いちばん大きい違い {difference:.6g}）",
        "en": "{object}: the weights of {count} vertices changed (largest difference {difference:.6g})",
    },

    # MARK: - 骨
    "versions.bones_added": {
        "ja": "{object}：骨が {count} 本増えました（{names}）",
        "en": "{object}: {count} bones were added ({names})",
    },
    "versions.bones_removed": {
        "ja": "{object}：骨が {count} 本無くなりました（{names}）",
        "en": "{object}: {count} bones were removed ({names})",
    },
    "versions.bones_moved": {
        "ja": "{object}：骨が {count} 本動きました（{names}。いちばん大きい動き {distance_mm:.3f} mm）",
        "en": "{object}: {count} bones moved ({names}; largest move {distance_mm:.3f} mm)",
    },
    "versions.bone_parents_changed": {
        "ja": "{object}：親が変わった骨が {count} 本あります（{names}）",
        "en": "{object}: the parent changed for {count} bones ({names})",
    },
    "versions.bone_deform_changed": {
        "ja": "{object}：形を動かすかどうか（use_deform）が変わった骨が {count} 本あります（{names}）",
        "en": "{object}: “deform” (use_deform) changed for {count} bones ({names})",
    },
    "versions.bone_order_changed": {
        "ja": "{object}：骨の並びが変わりました（書き出しの骨の番号 b000 なども変わります）",
        "en": "{object}: the bone order changed (the exported bone numbers such as b000 change too)",
    },

    # MARK: - ドライバー
    "versions.drivers_added": {
        "ja": "{object}：ドライバーが {count} 個増えました（{names}）",
        "en": "{object}: {count} drivers were added ({names})",
    },
    "versions.drivers_removed": {
        "ja": "{object}：ドライバーが {count} 個無くなりました（{names}）",
        "en": "{object}: {count} drivers were removed ({names})",
    },
    "versions.drivers_changed": {
        "ja": "{object}：中身（式・変数）が変わったドライバーが {count} 個あります（{names}）",
        "en": "{object}: {count} drivers changed (expression or variables) ({names})",
    },

    # MARK: - 材質
    "versions.material_slots_changed": {
        "ja": "{object}：材質の枠が変わりました（{old} → {new}）",
        "en": "{object}: the material slots changed ({old} → {new})",
    },
    "versions.face_materials_changed": {
        "ja": "{object}：材質が変わった面が {count} 個あります",
        "en": "{object}: the material changed on {count} faces",
    },

    # MARK: - モディファイア
    "versions.modifiers_added": {
        "ja": "{object}：モディファイアが {count} 個増えました（{names}）",
        "en": "{object}: {count} modifiers were added ({names})",
    },
    "versions.modifiers_removed": {
        "ja": "{object}：モディファイアが {count} 個無くなりました（{names}）",
        "en": "{object}: {count} modifiers were removed ({names})",
    },
    "versions.modifiers_type_changed": {
        "ja": "{object}：種類が変わったモディファイアが {count} 個あります（{names}）",
        "en": "{object}: the type changed for {count} modifiers ({names})",
    },
    "versions.modifiers_render_changed": {
        "ja": "{object}：描画で使うかどうかが変わったモディファイアが {count} 個あります（{names}）",
        "en": "{object}: “use in render” changed for {count} modifiers ({names})",
    },
    "versions.modifier_order_changed": {
        "ja": "{object}：モディファイアの並びが変わりました（{names}）",
        "en": "{object}: the order of the modifiers changed ({names})",
    },

    # MARK: - まとめ（信号の理由）
    "versions.summary_none": {
        "ja": "前の版から変わった所はありません",
        "en": "Nothing changed since the previous version",
    },
    "versions.summary_allowed": {
        "ja": "変えてよい所の外の変化はありません（変えてよい所の中の変化が {allowed} 件）",
        "en": "No change outside the allowed changes ({allowed} changes inside the allowed changes)",
    },
    "versions.summary_outside": {
        "ja": "変えてよい所の外の変化が {count} 件あります（変えてよい所の中の変化は {allowed} 件）",
        "en": "{count} changes are outside the allowed changes ({allowed} changes inside the allowed changes)",
    },
    "versions.no_previous": {
        "ja": "比べる前の版がありません（まだ版を保存していません）",
        "en": "There is no previous version to compare with (no version has been saved yet)",
    },
    "versions.snapshot_missing": {
        "ja": "版 {name} の記録（形・骨・ウェイトなどの控え）が見つからないか、読めません：{error}",
        "en": "The record (copy of shapes, bones, weights, etc.) of version {name} is missing or unreadable: {error}",
    },

    # MARK: - 保存と版の一覧
    "versions.saved": {
        "ja": "新しい版 {name} を保存しました（いま開いているファイルは、そのままです）",
        "en": "Saved the new version {name} (the file you have open stays the same)",
    },
    "versions.version_entry": {"ja": "第 {number} 版：{name}", "en": "Version {number}: {name}"},
}
