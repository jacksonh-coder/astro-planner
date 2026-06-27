from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

st.set_page_config(page_title='Astro Planner', page_icon='🌌', layout='wide')

CSV_PATH = 'data/messier.csv'
DEFAULT_LAT = 37.2636
DEFAULT_LON = 127.0286
DEFAULT_ALT_M = 40.0
DEFAULT_TZ = 9


def load_catalog(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f'파일을 찾을 수 없습니다: {path}')

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ['Name', 'RA', 'Dec', 'Magnitude']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'필수 컬럼이 없습니다: {missing}. CSV는 Name,RA,Dec,Magnitude 형식이어야 합니다.')

    df['Name'] = df['Name'].astype(str).str.strip()
    df['RA'] = df['RA'].astype(str).str.strip()
    df['Dec'] = df['Dec'].astype(str).str.strip()
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')

    coords = SkyCoord(df['RA'].tolist(), df['Dec'].tolist(), unit=(u.hourangle, u.deg), frame='icrs')
    df['ra_deg'] = coords.ra.deg
    df['dec_deg'] = coords.dec.deg

    df = df.dropna(subset=['Name', 'ra_deg', 'dec_deg']).drop_duplicates('Name').reset_index(drop=True)
    return df


def build_times(local_date: date, tz_offset_hours: int, start_hour: float, end_hour: float, step_min: int):
    base = datetime.combine(local_date, dtime(0, 0, 0))
    local_datetimes = [base + timedelta(minutes=m) for m in range(int(start_hour * 60), int(end_hour * 60) + 1, step_min)]
    utc_datetimes = [dt - timedelta(hours=tz_offset_hours) for dt in local_datetimes]
    return local_datetimes, Time(utc_datetimes)


def contiguous_spans(indices):
    if len(indices) == 0:
        return []
    spans = []
    s = p = indices[0]
    for i in indices[1:]:
        if i == p + 1:
            p = i
        else:
            spans.append((s, p))
            s = p = i
    spans.append((s, p))
    return spans


def analyze_targets(catalog, local_datetimes, times_astropy, location, min_alt_deg, az_min, az_max, mag_limit):
    frame = AltAz(obstime=times_astropy, location=location)
    rows = []
    paths = {}

    catalog2 = catalog.copy()
    catalog2 = catalog2[catalog2['Magnitude'].fillna(99) <= mag_limit]

    for _, rec in catalog2.iterrows():
        coord = SkyCoord(ra=rec['ra_deg'] * u.deg, dec=rec['dec_deg'] * u.deg, frame='icrs')
        altaz = coord.transform_to(frame)
        alt = np.array(altaz.alt.deg)
        az = np.array(altaz.az.deg)

        if az_min <= az_max:
            az_ok = (az >= az_min) & (az <= az_max)
        else:
            az_ok = (az >= az_min) | (az <= az_max)

        visible = (alt >= min_alt_deg) & az_ok
        paths[rec['Name']] = pd.DataFrame({
            'datetime_local': local_datetimes,
            'alt_deg': alt,
            'az_deg': az,
            'visible': visible,
            'r': 90 - alt,
        })

        if visible.any():
            idx = np.where(visible)[0]
            spans = contiguous_spans(idx)
            i0, i1 = max(spans, key=lambda ab: ab[1] - ab[0])
            imax = int(np.argmax(alt))
            rows.append({
                'name': rec['Name'],
                'magnitude': rec['Magnitude'],
                'max_alt_deg': round(float(np.max(alt)), 1),
                'start_time': local_datetimes[i0].strftime('%Y-%m-%d %H:%M'),
                'end_time': local_datetimes[i1].strftime('%Y-%m-%d %H:%M'),
                'best_time': local_datetimes[imax].strftime('%Y-%m-%d %H:%M'),
                'visible_hours': round((local_datetimes[i1] - local_datetimes[i0]).total_seconds() / 3600, 2),
                'start_az_deg': round(float(az[i0]), 1),
                'end_az_deg': round(float(az[i1]), 1),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(['max_alt_deg', 'visible_hours', 'magnitude'], ascending=[False, False, True])
    return result, paths


def make_polar_chart(selected_names, paths):
    fig = go.Figure()
    palette = ['#4f98a3', '#e8af34', '#6daa45', '#a86fdf', '#dd6974', '#fdab43', '#5591c7', '#d163a7']

    for i, name in enumerate(selected_names):
        p = paths[name]
        pv = p[p['visible']].copy()
        if pv.empty:
            continue
        color = palette[i % len(palette)]
        hover_time = [dt.strftime('%m-%d %H:%M') for dt in pv['datetime_local']]

        fig.add_trace(go.Scatterpolar(
            r=pv['r'],
            theta=pv['az_deg'],
            mode='lines',
            name=name,
            line=dict(width=2, color=color),
            customdata=np.stack([pv['alt_deg'], pv['az_deg'], hover_time], axis=-1),
            hovertemplate='<b>%{fullData.name}</b><br>시간: %{customdata[2]}<br>고도: %{customdata[0]:.1f}°<br>방위각: %{customdata[1]:.1f}°<extra></extra>'
        ))
        start = pv.iloc[0]
        end = pv.iloc[-1]
        fig.add_trace(go.Scatterpolar(
            r=[start['r'], end['r']],
            theta=[start['az_deg'], end['az_deg']],
            mode='markers+text',
            text=['start', 'end'],
            textposition='top center',
            marker=dict(size=8, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        height=760,
        paper_bgcolor='#0b1220',
        plot_bgcolor='#0b1220',
        font=dict(color='#e5eef7'),
        margin=dict(l=30, r=140, t=60, b=30),
        legend=dict(orientation='v', x=1.02, y=1.0),
        title='촬영 가능 시간대의 천구 이동 경로',
        polar=dict(
            bgcolor='#0b1220',
            angularaxis=dict(
                direction='clockwise',
                rotation=90,
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                gridcolor='rgba(255,255,255,0.15)',
                linecolor='rgba(255,255,255,0.25)'
            ),
            radialaxis=dict(
                range=[90, 0],
                tickmode='array',
                tickvals=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                ticktext=['80°', '70°', '60°', '50°', '40°', '30°', '20°', '10°', '0°'],
                gridcolor='rgba(255,255,255,0.15)',
                linecolor='rgba(255,255,255,0.25)'
            )
        )
    )
    return fig


@st.cache_data(show_spinner=False)
def cached_load_catalog(path: str):
    return load_catalog(path)


st.title('🌌 Astro Planner Dashboard')
st.caption('CSV 형식의 messier.csv(Name,RA,Dec,Magnitude)를 읽어 촬영 가능 대상과 천구 경로를 표시합니다.')

with st.sidebar:
    st.header('관측 설정')
    csv_path = st.text_input('CSV 경로', value=CSV_PATH)
    obs_date = st.date_input('관측 날짜', value=date.today())
    lat = st.number_input('위도', value=DEFAULT_LAT, format='%.6f')
    lon = st.number_input('경도', value=DEFAULT_LON, format='%.6f')
    altitude_m = st.number_input('해발고도(m)', value=DEFAULT_ALT_M, format='%.1f')
    tz_offset = st.number_input('UTC 오프셋', value=DEFAULT_TZ)

    st.divider()
    st.subheader('촬영 조건')
    min_alt_deg = st.slider('최소 고도', 0, 80, 25, 1)
    start_hour, end_hour = st.slider('로컬 시간 범위', 0.0, 24.0, (19.0, 23.5), 0.5)
    if end_hour <= start_hour:
        end_hour += 24
    az_min, az_max = st.slider('방위각 범위', 0, 359, (0, 359), 1)
    mag_limit = st.slider('최대 등급(작을수록 밝음)', -2.0, 15.0, 8.0, 0.1)
    step_min = st.selectbox('샘플 간격(분)', [2, 3, 5, 10, 15], index=2)
    top_n = st.slider('기본 표시 대상 수', 1, 30, 8, 1)

try:
    catalog = cached_load_catalog(csv_path)
except Exception as e:
    st.error(str(e))
    st.stop()

location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=altitude_m * u.m)
local_datetimes, times_astropy = build_times(obs_date, int(tz_offset), start_hour, end_hour, step_min)
results, paths = analyze_targets(catalog, local_datetimes, times_astropy, location, min_alt_deg, az_min, az_max, mag_limit)

left, mid, right = st.columns(3)
left.metric('카탈로그 대상 수', len(catalog))
mid.metric('등급 필터 후 대상 수', int((catalog['Magnitude'].fillna(99) <= mag_limit).sum()))
right.metric('촬영 가능 대상 수', len(results))

with st.expander('카탈로그 미리보기'):
    st.dataframe(catalog, use_container_width=True, hide_index=True)

if results.empty:
    st.warning('현재 조건에서 촬영 가능한 대상이 없습니다. 시간 범위, 최소 고도, 등급 제한을 조정해 보세요.')
    st.stop()

names = results['name'].tolist()
default_selected = names[:min(top_n, len(names))]

st.subheader('촬영 가능 대상 리스트')
selected_names = st.multiselect('차트에 표시할 대상', names, default=default_selected)
filtered = results[results['name'].isin(selected_names)] if selected_names else results.head(top_n)
st.dataframe(filtered, use_container_width=True, hide_index=True)

st.subheader('천구 경로')
fig = make_polar_chart(filtered['name'].tolist(), paths)
st.plotly_chart(fig, use_container_width=True)

st.subheader('대상별 상세 변화')
selected_detail = st.selectbox('상세 대상', filtered['name'].tolist())
detail = paths[selected_detail].copy()
detail['datetime_local'] = pd.to_datetime(detail['datetime_local'])

line_fig = go.Figure()
line_fig.add_trace(go.Scatter(
    x=detail['datetime_local'],
    y=detail['alt_deg'],
    mode='lines',
    name='Altitude',
    line=dict(color='#4f98a3', width=2)
))
line_fig.add_trace(go.Scatter(
    x=detail['datetime_local'],
    y=detail['az_deg'],
    mode='lines',
    name='Azimuth',
    yaxis='y2',
    line=dict(color='#e8af34', width=2)
))
line_fig.add_hline(y=min_alt_deg, line_dash='dash', line_color='rgba(255,255,255,0.35)')
line_fig.update_layout(
    height=420,
    paper_bgcolor='#0b1220',
    plot_bgcolor='#0b1220',
    font=dict(color='#e5eef7'),
    xaxis=dict(title='Local time'),
    yaxis=dict(title='Altitude (deg)', range=[0, 90]),
    yaxis2=dict(title='Azimuth (deg)', overlaying='y', side='right', range=[0, 360]),
    margin=dict(l=30, r=30, t=30, b=30),
    legend=dict(orientation='h', y=1.05, x=0)
)
st.plotly_chart(line_fig, use_container_width=True)

st.markdown('### 실행 방법')
st.code('streamlit run app.py', language='bash')