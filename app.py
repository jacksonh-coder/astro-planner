import streamlit as st
import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import plotly.graph_objects as go

데이터 불러오기
df = pd.read_csv("data/messier.csv")

Streamlit UI
st.title("Messier 천체 촬영 플래너")

사용자 입력
date = st.date_input("날짜 선택")
latitude = st.number_input("위도 입력 (예: 37.3)", value=37.3)
longitude = st.number_input("경도 입력 (예: 127.0)", value=127.0)
az_min = st.slider("방위각 최소", 0, 360, 90)
az_max = st.slider("방위각 최대", 0, 360, 200)
alt_min = st.slider("최소 고도", 0, 90, 30)

관측지 설정
location = EarthLocation(lat=latitudeu.deg, lon=longitudeu.deg)

시간 범위 설정
times = Time(f"{date} 20:00") + (u.hour * range(0, 10))

visible_targets = []

for _, row in df.iterrows():
coord = SkyCoord(row["RA"], row["Dec"], unit=(u.hourangle, u.deg))
altaz = coord.transform_to(AltAz(obstime=times, location=location))
altitudes = altaz.alt.deg
azimuths = altaz.az.deg

# 조건 필터링
if any((alt > alt_min) and (az_min <= az <= az_max) for alt, az in zip(altitudes, azimuths)):
visible_targets.append((row["Name"], row["Magnitude"], altitudes, azimuths, times))

밝기 순 정렬
visible_targets.sort(key=lambda x: x[1])

리스트 출력
st.subheader("촬영 가능 대상")
for name, mag, altitudes, azimuths, times in visible_targets:
if st.button(f"{name} (등급 {mag})"):
fig = go.Figure()
fig.add_trace(go.Scatter(x=times.datetime, y=altitudes, mode="lines", name="고도"))
fig.update_layout(title=f"{name} 시간별 고도 변화", xaxis_title="시간", yaxis_title="고도(°)")
st.plotly_chart(fig)