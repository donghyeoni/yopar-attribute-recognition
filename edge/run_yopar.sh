#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

if pgrep -f jetson_par_sender.py >/dev/null; then
    echo "이미 실행 중이다. 두 개가 GPU를 나눠 갖지 못한다."
    echo
    pgrep -af jetson_par_sender.py
    echo
    read -rp "기존 것을 끄고 새로 실행할까? [y/N] " ans
    case "$ans" in
        [yY]*) pkill -f jetson_par_sender.py; sleep 2 ;;
        *) echo "취소했다."; read -rp "엔터를 누르면 창이 닫힌다. "; exit 0 ;;
    esac
fi

echo "== service 인상착의 검색 =="
echo "찾는 사람: 서버가 등록한 사건을 따른다 (query.txt는 쓰지 않는다)."
echo "  사건이 등록되면 화면 왼쪽 위에 그 문장이 그대로 뜬다."
echo "  아직 없으면 NO ACTIVE CASE 로 표시되고 감시만 한다."
echo "끝낼 땐 Ctrl+C."
echo

python3 scripts/jetson_par_sender.py
status=$?

echo
echo "== 종료됨 (exit $status) =="
read -rp "엔터를 누르면 창이 닫힌다. "
