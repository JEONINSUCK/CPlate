# OCR

단계별 문자 인식 프로젝트


# CPlate : Car Plate

차량의 번호 인식 엔진을 개발하는 프로젝트

차량에서 얻어지는 정보는 점차 확대 예정



# django web server

웹 소스는 cplate 사이트를 도메인으로 합니다

```bash

  # dev
  docker run -d \
    --name cplate-db \
    -h cplate-db \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=cplate \
  	postgres



  # prod : 옵셩 최종 타협 필요
  docker run -d \
    --name cplate-db \
    -h cplate-db \
    -e POSTGRES_PASSWORD=cplate \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v /custom/mount:/var/lib/postgresql/data \
  	postgres
  cplate

```

- 새로 설치 및 배포시 패키지 확인 필요

```bash
# save
pip3 freeze > requirements.txt

# install
pip install -r requirements.txt

```