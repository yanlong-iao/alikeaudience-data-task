#!/usr/bin/env python3
"""
AlikeAudience — Data Test Task 1
Enrichment & analytics pipeline for user event data
(user_id, timestamp, lat_long, ip_address, user_agent).

Design notes
------------
* Runs fully offline (no GeoIP / geocoding APIs): country & city are derived
  with a vectorized nearest-city lookup over a curated gazetteer of ~190
  cities across East / Southeast Asia (the data's bounding box). In
  production this would be swapped for MaxMind GeoIP2 + a Natural Earth
  point-in-polygon join — the interface below stays the same.
* The source column `lat_long` is actually LONGITUDE LATITUDE (verified:
  |col0| > 90 for 100% of rows). The pipeline fixes and documents this.
"""

import re
import json
import ipaddress
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. Gazetteer: (city, country, lat, lon) — major + secondary cities in bbox
# ----------------------------------------------------------------------------
CITIES = [
    # Indonesia
    ("Jakarta","Indonesia",-6.2088,106.8456),("Surabaya","Indonesia",-7.2575,112.7521),
    ("Bandung","Indonesia",-6.9175,107.6191),("Medan","Indonesia",3.5952,98.6722),
    ("Semarang","Indonesia",-6.9667,110.4167),("Makassar","Indonesia",-5.1477,119.4327),
    ("Palembang","Indonesia",-2.9761,104.7754),("Tangerang","Indonesia",-6.1783,106.6319),
    ("Depok","Indonesia",-6.4025,106.7942),("Bekasi","Indonesia",-6.2349,106.9896),
    ("Padang","Indonesia",-0.9471,100.4172),("Denpasar","Indonesia",-8.6500,115.2167),
    ("Malang","Indonesia",-7.9797,112.6304),("Samarinda","Indonesia",-0.5022,117.1536),
    ("Banjarmasin","Indonesia",-3.3194,114.5908),("Pontianak","Indonesia",-0.0263,109.3425),
    ("Manado","Indonesia",1.4748,124.8421),("Yogyakarta","Indonesia",-7.7956,110.3695),
    ("Pekanbaru","Indonesia",0.5071,101.4478),("Bandar Lampung","Indonesia",-5.4295,105.2610),
    ("Jayapura","Indonesia",-2.5330,140.7181),("Ambon","Indonesia",-3.6954,128.1814),
    ("Kupang","Indonesia",-10.1772,123.6070),("Mataram","Indonesia",-8.5833,116.1167),
    ("Balikpapan","Indonesia",-1.2379,116.8529),("Batam","Indonesia",1.0456,104.0305),
    ("Banda Aceh","Indonesia",5.5483,95.3238),("Jambi","Indonesia",-1.6101,103.6131),
    ("Surakarta","Indonesia",-7.5561,110.8317),("Palu","Indonesia",-0.8917,119.8707),
    ("Kendari","Indonesia",-3.9450,122.4989),("Sorong","Indonesia",-0.8762,131.2558),
    # Philippines
    ("Manila","Philippines",14.5995,120.9842),("Quezon City","Philippines",14.6760,121.0437),
    ("Cebu City","Philippines",10.3157,123.8854),("Davao City","Philippines",7.1907,125.4553),
    ("Zamboanga","Philippines",6.9214,122.0790),("Cagayan de Oro","Philippines",8.4542,124.6319),
    ("Iloilo City","Philippines",10.7202,122.5621),("Bacolod","Philippines",10.6770,122.9500),
    ("General Santos","Philippines",6.1164,125.1716),("Baguio","Philippines",16.4023,120.5960),
    ("Angeles","Philippines",15.1450,120.5887),("Naga","Philippines",13.6218,123.1948),
    ("Tacloban","Philippines",11.2444,125.0039),("Puerto Princesa","Philippines",9.7392,118.7353),
    ("Laoag","Philippines",18.1978,120.5936),("Tuguegarao","Philippines",17.6131,121.7269),
    # Malaysia
    ("Kuala Lumpur","Malaysia",3.1390,101.6869),("George Town","Malaysia",5.4141,100.3288),
    ("Johor Bahru","Malaysia",1.4927,103.7414),("Ipoh","Malaysia",4.5975,101.0901),
    ("Kuching","Malaysia",1.5533,110.3592),("Kota Kinabalu","Malaysia",5.9804,116.0735),
    ("Kuantan","Malaysia",3.8077,103.3260),("Kota Bharu","Malaysia",6.1254,102.2381),
    ("Alor Setar","Malaysia",6.1248,100.3678),("Miri","Malaysia",4.3995,113.9914),
    ("Sandakan","Malaysia",5.8402,118.1179),("Tawau","Malaysia",4.2448,117.8911),
    ("Sibu","Malaysia",2.2870,111.8305),
    # Singapore / Brunei / Timor-Leste
    ("Singapore","Singapore",1.3521,103.8198),
    ("Bandar Seri Begawan","Brunei",4.9031,114.9398),
    ("Dili","Timor-Leste",-8.5569,125.5603),
    # Thailand
    ("Bangkok","Thailand",13.7563,100.5018),("Chiang Mai","Thailand",18.7883,98.9853),
    ("Hat Yai","Thailand",7.0086,100.4747),("Khon Kaen","Thailand",16.4419,102.8360),
    ("Udon Thani","Thailand",17.4138,102.7877),("Nakhon Ratchasima","Thailand",14.9799,102.0978),
    ("Phuket","Thailand",7.8804,98.3923),("Pattaya","Thailand",12.9236,100.8825),
    ("Surat Thani","Thailand",9.1382,99.3217),("Ubon Ratchathani","Thailand",15.2287,104.8564),
    # Myanmar
    ("Yangon","Myanmar",16.8661,96.1951),("Mandalay","Myanmar",21.9588,96.0891),
    ("Naypyidaw","Myanmar",19.7633,96.0785),("Mawlamyine","Myanmar",16.4905,97.6283),
    ("Taunggyi","Myanmar",20.7892,97.0378),("Sittwe","Myanmar",20.1500,92.9000),
    ("Myitkyina","Myanmar",25.3833,97.4000),
    # Laos / Cambodia
    ("Vientiane","Laos",17.9757,102.6331),("Luang Prabang","Laos",19.8834,102.1347),
    ("Pakse","Laos",15.1202,105.7987),("Savannakhet","Laos",16.5566,104.7523),
    ("Phnom Penh","Cambodia",11.5564,104.9282),("Siem Reap","Cambodia",13.3671,103.8448),
    ("Battambang","Cambodia",13.0957,103.2022),("Sihanoukville","Cambodia",10.6104,103.5284),
    # Vietnam
    ("Ho Chi Minh City","Vietnam",10.8231,106.6297),("Hanoi","Vietnam",21.0285,105.8542),
    ("Da Nang","Vietnam",16.0544,108.2022),("Haiphong","Vietnam",20.8449,106.6881),
    ("Can Tho","Vietnam",10.0452,105.7469),("Hue","Vietnam",16.4637,107.5909),
    ("Nha Trang","Vietnam",12.2388,109.1967),("Vinh","Vietnam",18.6733,105.6922),
    ("Buon Ma Thuot","Vietnam",12.6667,108.0500),("Quy Nhon","Vietnam",13.7830,109.2196),
    # China
    ("Beijing","China",39.9042,116.4074),("Shanghai","China",31.2304,121.4737),
    ("Guangzhou","China",23.1291,113.2644),("Shenzhen","China",22.5431,114.0579),
    ("Chengdu","China",30.5728,104.0668),("Chongqing","China",29.4316,106.9123),
    ("Wuhan","China",30.5928,114.3055),("Xi'an","China",34.3416,108.9398),
    ("Hangzhou","China",30.2741,120.1551),("Nanjing","China",32.0603,118.7969),
    ("Tianjin","China",39.3434,117.3616),("Shenyang","China",41.8057,123.4315),
    ("Harbin","China",45.8038,126.5350),("Changchun","China",43.8171,125.3235),
    ("Dalian","China",38.9140,121.6147),("Qingdao","China",36.0671,120.3826),
    ("Jinan","China",36.6512,117.1201),("Zhengzhou","China",34.7466,113.6254),
    ("Changsha","China",28.2282,112.9388),("Kunming","China",24.8801,102.8329),
    ("Nanning","China",22.8170,108.3665),("Guiyang","China",26.6470,106.6302),
    ("Fuzhou","China",26.0745,119.2965),("Xiamen","China",24.4798,118.0894),
    ("Nanchang","China",28.6820,115.8579),("Hefei","China",31.8206,117.2272),
    ("Shijiazhuang","China",38.0428,114.5149),("Taiyuan","China",37.8706,112.5489),
    ("Lanzhou","China",36.0611,103.8343),("Xining","China",36.6171,101.7782),
    ("Hohhot","China",40.8424,111.7490),("Urumqi","China",43.8256,87.6168),
    ("Lhasa","China",29.6520,91.1721),("Haikou","China",20.0444,110.1920),
    ("Sanya","China",18.2528,109.5119),("Wenzhou","China",27.9938,120.6994),
    ("Suzhou","China",31.2989,120.5853),("Dongguan","China",23.0207,113.7518),
    ("Foshan","China",23.0215,113.1214),("Ningbo","China",29.8683,121.5440),
    # Taiwan / HK / Macau
    ("Taipei","Taiwan",25.0330,121.5654),("Kaohsiung","Taiwan",22.6273,120.3014),
    ("Taichung","Taiwan",24.1477,120.6736),("Tainan","Taiwan",22.9998,120.2269),
    ("Hualien","Taiwan",23.9769,121.6044),
    ("Hong Kong","Hong Kong",22.3193,114.1694),("Macau","Macau",22.1987,113.5439),
    # Japan
    ("Tokyo","Japan",35.6762,139.6503),("Yokohama","Japan",35.4437,139.6380),
    ("Osaka","Japan",34.6937,135.5023),("Nagoya","Japan",35.1815,136.9066),
    ("Sapporo","Japan",43.0618,141.3545),("Fukuoka","Japan",33.5904,130.4017),
    ("Kobe","Japan",34.6901,135.1956),("Kyoto","Japan",35.0116,135.7681),
    ("Sendai","Japan",38.2682,140.8694),("Hiroshima","Japan",34.3853,132.4553),
    ("Niigata","Japan",37.9162,139.0364),("Kanazawa","Japan",36.5613,136.6562),
    ("Okayama","Japan",34.6551,133.9195),("Kumamoto","Japan",32.8032,130.7079),
    ("Kagoshima","Japan",31.5966,130.5571),("Naha","Japan",26.2124,127.6809),
    ("Matsuyama","Japan",33.8392,132.7658),("Shizuoka","Japan",34.9756,138.3828),
    ("Hakodate","Japan",41.7687,140.7291),("Aomori","Japan",40.8222,140.7474),
    # South Korea
    ("Seoul","South Korea",37.5665,126.9780),("Busan","South Korea",35.1796,129.0756),
    ("Incheon","South Korea",37.4563,126.7052),("Daegu","South Korea",35.8714,128.6014),
    ("Daejeon","South Korea",36.3504,127.3845),("Gwangju","South Korea",35.1595,126.8526),
    ("Ulsan","South Korea",35.5384,129.3114),("Jeju City","South Korea",33.4996,126.5312),
    # North Korea / Mongolia / Russia (Far East) / PNG
    ("Pyongyang","North Korea",39.0392,125.7625),("Hamhung","North Korea",39.9183,127.5364),
    ("Chongjin","North Korea",41.7956,129.7758),
    ("Ulaanbaatar","Mongolia",47.8864,106.9057),
    ("Vladivostok","Russia",43.1155,131.8855),("Ussuriysk","Russia",43.8029,131.9458),
    ("Port Moresby","Papua New Guinea",-9.4438,147.1803),
    ("Vanimo","Papua New Guinea",-2.6819,141.3031),("Madang","Papua New Guinea",-5.2246,145.7967),
]

def build_geocoder():
    arr = np.array([(c[2], c[3]) for c in CITIES])
    names = [c[0] for c in CITIES]
    countries = [c[1] for c in CITIES]
    lat_r, lon_r = np.radians(arr[:, 0]), np.radians(arr[:, 1])

    def nearest(lat, lon, chunk=20000):
        """Vectorized haversine nearest-city; returns (city, country, km)."""
        idx_all, km_all = [], []
        for s in range(0, len(lat), chunk):
            la = np.radians(lat[s:s+chunk])[:, None]
            lo = np.radians(lon[s:s+chunk])[:, None]
            dphi = la - lat_r[None, :]
            dlmb = lo - lon_r[None, :]
            h = np.sin(dphi/2)**2 + np.cos(la)*np.cos(lat_r[None, :])*np.sin(dlmb/2)**2
            d = 2*6371.0*np.arcsin(np.sqrt(h))
            i = d.argmin(axis=1)
            idx_all.append(i)
            km_all.append(d[np.arange(len(i)), i])
        idx = np.concatenate(idx_all); km = np.concatenate(km_all)
        return ([names[i] for i in idx], [countries[i] for i in idx], km)
    return nearest

# ----------------------------------------------------------------------------
# 2. User-agent parser (regex-based, offline)
# ----------------------------------------------------------------------------
BRAND_RULES = [
    (r'^SM-|^GT-|^Galaxy', 'Samsung'), (r'^Redmi|^POCO|^Mi[ _]|^MI[ _]|^M2\d{3}|^2\d{3}[0-9A-Z]{6,}|^Xiaomi|^21\d{6,}', 'Xiaomi'),
    (r'^CPH\d|^OPPO|^PBA|^PCT|^PDV|^RMX(?=.*OPPO)', 'OPPO'), (r'^RMX\d', 'Realme'),
    (r'^vivo|^V2\d{3}', 'vivo'), (r'^moto|^Moto|^XT\d{4}', 'Motorola'),
    (r'^Pixel', 'Google'), (r'^Nokia', 'Nokia'), (r'^HUAWEI|^(ANE|ELE|VOG|MAR|STK|JKM|POT|LYA|EML|CLT|COL|PRA|FIG|SNE|INE|ATU|AGS|MRD|DUB|AMN|KSA|JAT|STF|WAS|BLA|NEO|EVR|TAS|VTR)-', 'Huawei'),
    (r'^(JSN|HRY|KSE|STK|COR|BKK|LLD|DUA)-|^Honor|^HONOR', 'Honor'),
    (r'^Infinix', 'Infinix'), (r'^TECNO', 'Tecno'), (r'^itel', 'itel'),
    (r'^Lenovo|^TB-', 'Lenovo'), (r'^ASUS|^ZS\d|^ZB\d', 'Asus'),
    (r'^ONEPLUS|^OnePlus|^(IN2|KB2|LE2|HD19|GM19|NE2|DE2|BE2|DN2|EB2|CPH24)\d*', 'OnePlus'),
    (r'^REVVL|^TMAF|^TMRV', 'T-Mobile'), (r'^WTCELERO|^Celero', 'Wiko/Celero'),
    (r'^B131DL|^A[0-9]{3}DL|^TCL|^T[0-9]{3}[A-Z]{2}', 'TCL'), (r'^EC211|^Cricket', 'Cricket/EC'),
    (r'^LM-|^LG-|^LGM', 'LG'), (r'^SonyEricsson|^Xperia|^[A-Z]{1,2}-?\d{4}(?=.*Sony)|^XQ-', 'Sony'),
    (r'^HTC', 'HTC'), (r'^Meizu', 'Meizu'), (r'^ZTE|^Blade', 'ZTE'),
    (r'^SHARP|^SH-', 'Sharp'), (r'^KYOCERA|^KYV', 'Kyocera'),
    (r'^Wiko', 'Wiko'), (r'^Alcatel|^\d{4}[A-Z](?=.*Alcatel)', 'Alcatel'),
]

def parse_ua(ua):
    """Return dict of device/OS/browser features from a raw UA string."""
    out = dict(os_family=None, os_version=None, browser=None, browser_version=None,
               device_brand=None, device_model=None, form_factor=None, is_webview=False)
    if not isinstance(ua, str) or not ua.strip():
        return out
    # --- OS ---
    m = re.search(r'iPhone OS (\d+[._]\d+)', ua) or re.search(r'CPU OS (\d+[._]\d+)', ua)
    if 'iPhone' in ua or 'iPad' in ua:
        out['os_family'] = 'iOS'
        out['os_version'] = m.group(1).replace('_', '.') if m else None
        out['device_brand'] = 'Apple'
        out['device_model'] = 'iPad' if 'iPad' in ua else 'iPhone'
        out['form_factor'] = 'tablet' if 'iPad' in ua else 'phone'
    elif 'Android' in ua:
        out['os_family'] = 'Android'
        m = re.search(r'Android (\d+(?:\.\d+)*)', ua)
        out['os_version'] = m.group(1) if m else None
        # model: "; <model> Build/" or last token before ")"
        m = re.search(r';\s*([^;)]+?)\s+Build/', ua)
        if not m:
            m = re.search(r'Android [\d.]+;\s*([^;)]+)\)', ua)
        model = m.group(1).strip() if m else None
        if model and model.lower() not in ('k', 'wv'):
            out['device_model'] = model
            for pat, brand in BRAND_RULES:
                if re.search(pat, model):
                    out['device_brand'] = brand
                    break
        out['form_factor'] = 'tablet' if ('Tablet' in ua or (model and re.search(r'Tab|Pad', model or ''))) \
                             else ('phone' if 'Mobile' in ua else 'tablet')
    elif 'Windows NT' in ua:
        out['os_family'] = 'Windows'
        m = re.search(r'Windows NT (\d+\.\d+)', ua)
        nt = m.group(1) if m else None
        out['os_version'] = {'10.0': '10', '6.3': '8.1', '6.2': '8', '6.1': '7'}.get(nt, nt)
        out['form_factor'] = 'desktop'
    elif 'Macintosh' in ua:
        out['os_family'] = 'macOS'
        m = re.search(r'Mac OS X (\d+[._]\d+(?:[._]\d+)?)', ua)
        out['os_version'] = m.group(1).replace('_', '.') if m else None
        out['device_brand'] = 'Apple'
        out['form_factor'] = 'desktop'
    elif 'Linux' in ua or 'X11' in ua:
        out['os_family'] = 'Linux'
        out['form_factor'] = 'desktop'
    # --- Browser ---
    out['is_webview'] = ('; wv)' in ua) or bool(re.search(r'Version/4\.0.*Chrome', ua))
    for pat, name in [
        (r'SamsungBrowser/([\d.]+)', 'Samsung Internet'), (r'UCBrowser/([\d.]+)', 'UC Browser'),
        (r'MiuiBrowser/([\d.]+)', 'Miui Browser'), (r'EdgA?/([\d.]+)', 'Edge'),
        (r'OPR/([\d.]+)', 'Opera'), (r'Firefox/([\d.]+)', 'Firefox'),
        (r'CriOS/([\d.]+)', 'Chrome iOS'), (r'FxiOS/([\d.]+)', 'Firefox iOS'),
        (r'Chrome/([\d.]+)', 'Chrome'),
    ]:
        m = re.search(pat, ua)
        if m:
            out['browser'], out['browser_version'] = name, m.group(1).split('.')[0]
            break
    else:
        if 'Safari' in ua:
            m = re.search(r'Version/([\d.]+)', ua)
            out['browser'] = 'Safari'
            out['browser_version'] = m.group(1).split('.')[0] if m else None
        elif re.search(r'AppleWebKit.*Mobile/\w+$', ua):
            out['browser'] = 'iOS WebView'
            out['is_webview'] = True
    if out['is_webview'] and out['browser'] in ('Chrome', None):
        out['browser'] = 'Android WebView' if out['os_family'] == 'Android' else out['browser']
    return out

# ----------------------------------------------------------------------------
# 3. IP features (stdlib only)
# ----------------------------------------------------------------------------
def ip_features(ip):
    try:
        a = ipaddress.ip_address(ip)
        return a.version, a.is_global, (
            'private' if a.is_private else
            'reserved' if a.is_reserved else
            'multicast' if a.is_multicast else
            'link_local' if a.is_link_local else 'public')
    except ValueError:
        return 0, False, 'invalid'

# ----------------------------------------------------------------------------
# 4. Pipeline
# ----------------------------------------------------------------------------
def run(src, outdir):
    outdir = Path(outdir); outdir.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(src)
    n = len(df)

    # -- timestamps --
    df['ts'] = pd.to_datetime(df['timestamp'], utc=True)
    df['date'] = df['ts'].dt.date.astype(str)
    df['hour_utc'] = df['ts'].dt.hour
    df['dow'] = df['ts'].dt.day_name()
    df['is_weekend'] = df['ts'].dt.dayofweek >= 5

    # -- coordinates (source column is LON LAT, despite its name) --
    ll = df['lat_long'].str.split(' ', expand=True).astype(float)
    df['longitude'], df['latitude'] = ll[0], ll[1]
    assert (df['latitude'].abs() <= 90).all() and (df['longitude'].abs() <= 180).all()

    print('reverse geocoding...')
    nearest = build_geocoder()
    city, country, km = nearest(df['latitude'].values, df['longitude'].values)
    df['nearest_city'] = city
    df['country'] = country
    df['km_to_city'] = km.round(1)
    # low-confidence flag: far from any known city (ocean / remote)
    df['geo_confidence'] = np.where(km < 100, 'high', np.where(km < 300, 'medium', 'low'))

    # local time from longitude-approximated timezone offset (production: tz polygons)
    tz_offset = np.round(df['longitude'] / 15.0).astype(int)
    df['local_hour'] = (df['hour_utc'] + tz_offset) % 24
    df['local_part_of_day'] = pd.cut(df['local_hour'], [-1, 5, 11, 17, 21, 24],
                                     labels=['night', 'morning', 'afternoon', 'evening', 'night2'])
    df['local_part_of_day'] = df['local_part_of_day'].replace('night2', 'night')

    # -- IP --
    print('classifying IPs...')
    ipf = df['ip_address'].map(ip_features)
    df['ip_version'] = [x[0] for x in ipf]
    df['ip_routable'] = [x[1] for x in ipf]
    df['ip_scope'] = [x[2] for x in ipf]

    # -- user agent --
    print('parsing user agents (cached by unique UA)...')
    uniq = {ua: parse_ua(ua) for ua in df['user_agent'].dropna().unique()}
    empty = parse_ua(None)
    feats = df['user_agent'].map(lambda x: uniq.get(x, empty))
    for k in empty:
        df[k] = [f[k] for f in feats]
    df['ua_missing'] = df['user_agent'].isna()

    # -- write enriched dataset --
    keep = ['user_id', 'timestamp', 'latitude', 'longitude', 'country', 'nearest_city',
            'km_to_city', 'geo_confidence', 'local_hour', 'local_part_of_day', 'date',
            'hour_utc', 'dow', 'is_weekend', 'ip_address', 'ip_version', 'ip_routable',
            'ip_scope', 'os_family', 'os_version', 'device_brand', 'device_model',
            'form_factor', 'browser', 'browser_version', 'is_webview', 'ua_missing']
    df[keep].to_csv(outdir / 'enriched_data.csv', index=False)

    # -- summary stats --
    stats = {
        'rows': n,
        'unique_users': int(df['user_id'].nunique()),
        'date_range': [str(df['ts'].min()), str(df['ts'].max())],
        'missing_user_agent': int(df['ua_missing'].sum()),
        'non_routable_ips': int((~df['ip_routable']).sum()),
        'ip_version_split': df['ip_version'].value_counts().to_dict(),
        'top_countries': df['country'].value_counts().head(12).to_dict(),
        'os_split': df['os_family'].value_counts(dropna=False).head(8).to_dict(),
        'top_brands': df['device_brand'].value_counts().head(12).to_dict(),
        'top_models': df['device_model'].value_counts().head(12).to_dict(),
        'browser_split': df['browser'].value_counts().head(8).to_dict(),
        'webview_share': float(df['is_webview'].mean()),
        'form_factor': df['form_factor'].value_counts(dropna=False).to_dict(),
        'geo_confidence': df['geo_confidence'].value_counts().to_dict(),
        'events_by_date': df.groupby('date').size().to_dict(),
        'events_by_local_hour': df.groupby('local_hour').size().to_dict(),
    }
    with open(outdir / 'summary_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # -- charts --
    print('rendering charts...')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    df['country'].value_counts().head(10).sort_values().plot.barh(ax=axes[0,0], color='#4C8BF5')
    axes[0,0].set_title('Events by country (top 10)')
    df.groupby('date').size().plot(ax=axes[0,1], color='#4C8BF5')
    axes[0,1].set_title('Daily volume (Dec 2022)'); axes[0,1].tick_params(axis='x', rotation=45, labelsize=7)
    df.groupby('local_hour').size().plot.bar(ax=axes[0,2], color='#4C8BF5')
    axes[0,2].set_title('Events by local hour (longitude-approx.)')
    df['device_brand'].value_counts().head(10).sort_values().plot.barh(ax=axes[1,0], color='#F5804C')
    axes[1,0].set_title('Device brand (top 10)')
    df['os_family'].value_counts().plot.pie(ax=axes[1,1], autopct='%1.1f%%', ylabel='')
    axes[1,1].set_title('OS split')
    df['browser'].value_counts().head(8).sort_values().plot.barh(ax=axes[1,2], color='#3CB371')
    axes[1,2].set_title('Browser / WebView (top 8)')
    plt.tight_layout()
    plt.savefig(outdir / 'analytics_charts.png', dpi=130)

    # geo scatter
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df['longitude'], df['latitude'], s=1, alpha=0.15, c='#4C8BF5')
    ax.set_title('Event locations (100k points) — East & Southeast Asia')
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
    plt.tight_layout()
    plt.savefig(outdir / 'geo_scatter.png', dpi=130)

    print('done.')
    return stats

if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'alikeaudience_data_test.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else '.'
    s = run(src, out)
    print(json.dumps({k: s[k] for k in ('rows', 'top_countries', 'os_split', 'top_brands')},
                     indent=2, default=str))
