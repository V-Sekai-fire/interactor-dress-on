// godot-lite: Node/Node3D as plain Objects. There is no scene tree, so no
// notification is ever delivered; a Node3D subclass is driven by direct
// calls. The sketcher works in local coordinates and reads no transform.
#pragma once

#include "core/object/class_db.h"
#include "core/object/object.h"

namespace gdl {

class Node : public Object {
	GDCLASS(Node, Object);

public:
	enum {
		NOTIFICATION_ENTER_TREE = 10,
	};

	Node() = default;
};

class Node3D : public Node {
	GDCLASS(Node3D, Node);

public:
	Node3D() = default;
};

} // namespace gdl
