#!/bin/sh
# Clones the upstream sources ottis is checked against into third_party/
# (gitignored). Nothing builds from them; they are the reference for porting
# and the Python oracle (PLAN.md). Pinned so a fresh clone matches the
# commit PLAN.md's research was done against.
set -eu

cd "$(dirname "$0")/.."
mkdir -p third_party

fetch() { # dir url commit
	if [ -d "third_party/$1/.git" ]; then
		echo "third_party/$1: present ($(git -C "third_party/$1" rev-parse --short HEAD))"
		return
	fi
	git init -q "third_party/$1"
	git -C "third_party/$1" remote add origin "$2"
	git -C "third_party/$1" fetch -q --depth 1 origin "$3"
	git -C "third_party/$1" checkout -q --detach FETCH_HEAD
	echo "third_party/$1: $3"
}

fetch pocket-tts https://github.com/kyutai-labs/pocket-tts.git b94db177bc0f77517d9b124843e25605d6f9aed8
