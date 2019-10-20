#!/bin/sh

# to enable:
# cp pre-commit.sh .git/hooks/pre-commit
# chmod +x .git/hooks/pre-commit

git diff --staged --name-only HEAD | grep -E '\.py$' | while read -r f; do
    if ! pipenv run flake8 --max-line-length=88 --ignore=E203,W503 "$f"; then
        echo "Please fix errors before commiting";
        exit 1;
    fi
done
