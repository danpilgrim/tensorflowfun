#!/bin/sh
# HOW TO USE ./gitpush "message"(in quotes)
# Created by Daniel Pilgrim github.com/danpilgrim
message="$1"

git add --all
git commit -m $message
git push
