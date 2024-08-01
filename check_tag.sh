#! /bin/bash

if [[ -n "$1" ]]; then
    if ! grep "$1" "$2"; then
        echo Tag $1 does not match setup.py version. Bail.
        exit 1
    fi
fi
exit 0
