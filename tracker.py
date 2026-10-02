import json
import requests
import numpy as np
import scipy.ndimage
from skimage import measure
from PIL import Image
from io import BytesIO
import math

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"

LAZIO_HUBS = {
    "Roma": [41.9028, 12.4964],
    "Latina": [41.4676, 12.9036],
    "Frosinone": [41.6425, 13.3486],
    "Viterbo": [42.4172, 12.1081],
    "Rieti": [42.4048, 12.8631],
    "Terminillo": [42.4725, 12.9814]
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def tile_to_lat_lon(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    return math.degrees(lat_rad), lon_deg

def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile

def extract_cells_from_path(host, radar_path, zoom, x_min, x_max, y_min, y_max, tile_size):
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    stitched_image = Image.new("RGBA", (width, height))
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_url = f"{host}{radar_path}/{tile_size}/{zoom}/{x}/{y}/2/1_1.png"
            try:
                r = requests.get(tile_url, timeout=4)
                if r.status_code == 200:
                    t_img = Image.open(BytesIO(r.content)).convert("RGBA")
                    stitched_image.paste(t_img, ((x - x_min) * tile_size, (y - y_min) * tile_size))
            except Exception:
                continue

    img_array = np.array(stitched_image)
    alpha_channel = img_array[:, :, 3]
    precipitation_mask = alpha_channel > 150  
    closed_mask = scipy.ndimage.binary_closing(precipitation_mask, structure=np.ones((4, 4), dtype=bool), iterations=2)
    labeled_array, num_features = scipy.ndimage.label(closed_mask)
    
    cells = []
    if num_features > 0:
        objects = scipy.ndimage.find_objects(labeled_array)
        for i, slc in enumerate(objects):
            if slc is None:
                continue
            sub_mask = (labeled_array[slc] == (i + 1))
            pixel_count = np.sum(sub_mask)
            if pixel_count < 200: 
                continue
                
            cy_local, cx_local = scipy.ndimage.center_of_mass(sub_mask)
            lat_c, lon_c = tile_to_lat_lon(x_min + (slc[1].start + cx_local)/tile_size, 
                                           y_min + (slc[0].start + cy_local)/tile_size, zoom)
            
            if 41.0 <= lat_c <= 43.2 and 12.0 <= lon_c <= 14.2:
                padded_mask = np.pad(sub_mask, pad_width=1, mode='constant', constant_values=0)
                contours = measure.find_contours(padded_mask.astype(float), 0.5)
                poly_coords = []
                if contours:
                    contour = max(contours, key=len)
                    for pt in contour:
                        p_y = slc[0].start + pt[0] - 1
                        p_x = slc[1].start + pt[1] - 1
                        lat, lon = tile_to_lat_lon(x_min + (p_x / tile_size), y_min + (p_y / tile_size), zoom)
                        poly_coords.append([round(lon, 4), round(lat, 4)])
                    if len(poly_coords) >= 3 and poly_coords[0] != poly_coords[-1]:
                        poly_coords.append(poly_coords[0])

                cells.append({
                    "center": [round(lat_c, 4), round(lon_c, 4)],
                    "pixel_count": int(pixel_count),
                    "polygon": poly_coords
                })
    return cells

def main():
    try:
        resp = requests.get(RAINVIEWER_API, timeout=15)
        data = resp.json()
        host = data.get("host", "https://tilecache.rainviewer.com")
        past_radar = data.get("radar", {}).get("past", [])
        
        if not past_radar:
            raise Exception("Nessun dato radar trovato.")
            
        zoom = 7
        x_min_f, y_max_f = lat_lon_to_tile(41.0, 12.0, zoom)
        x_max_f, y_min_f = lat_lon_to_tile(43.2, 14.2, zoom)
        x_min, x_max = int(math.floor(x_min_f)), int(math.ceil(x_max_f))
        y_min, y_max = int(math.floor(y_min_f)), int(math.ceil(y_max_f))
        tile_size = 512

        selected_past = past_radar[-3:] if len(past_radar) >= 3 else past_radar
        history_frames_cells = []
        for frame in selected_past:
            history_frames_cells.append(extract_cells_from_path(host, frame["path"], zoom, x_min, x_max, y_min, y_max, tile_size))

        features = []
        latest_cells = history_frames_cells[-1] if history_frames_cells else []
        prev_cells = history_frames_cells[-2] if len(history_frames_cells) >= 2 else []

        for idx, current_cell in enumerate(latest_cells):
            c_lat, c_lon = current_cell["center"]
            pixel_count = current_cell["pixel_count"]
            
            path_coords = [[c_lon, c_lat]]
            prev_pixel_count = pixel_count
            
            if prev_cells:
                closest_prev = min(prev_cells, key=lambda c: math.sqrt((c["center"][0]-c_lat)**2 + (c["center"][1]-c_lon)**2))
                dist_prev = math.sqrt((closest_prev["center"][0]-c_lat)**2 + (closest_prev["center"][1]-c_lon)**2)
                if dist_prev < 0.5:
                    path_coords.insert(0, [closest_prev["center"][1], closest_prev["center"][0]])
                    prev_pixel_count = closest_prev["pixel_count"]

            if len(path_coords) >= 2:
                dlat = path_coords[-1][1] - path_coords[-2][1]
                dlon = path_coords[-1][0] - path_coords[-2][0]
                dist_km = math.sqrt(dlat**2 + dlon**2) * 111.0
                speed_kmh = round(dist_km / (10.0 / 60.0), 1)
            else:
                dlat, dlon = 0.05, 0.05
                speed_kmh = 35.0

            if speed_kmh < 5.0:
                speed_kmh = 5.0

            forecast_lon = round(c_lon + (dlon * 3), 4)
            forecast_lat = round(c_lat + (dlat * 3), 4)
            forecast_coords = [[c_lon, c_lat], [forecast_lon, forecast_lat]]

            eta_reports = []
            for hub_name, (h_lat, h_lon) in LAZIO_HUBS.items():
                dist_to_hub = haversine(c_lat, c_lon, h_lat, h_lon)
                bearing_cell = math.atan2(dlon, dlat)
                bearing_hub = math.atan2(h_lon - c_lon, h_lat - c_lat)
                if abs(bearing_cell - bearing_hub) < math.radians(90) and dist_to_hub < 120:
                    hours = dist_to_hub / speed_kmh
                    minutes = int(hours * 60)
                    eta_reports.append(f"{hub_name}: ~{minutes} min")
            eta_string = ", ".join(eta_reports[:2]) if eta_reports else "Nessun target imminente"

            forecast_dist_km = haversine(c_lat, c_lon, forecast_lat, forecast_lon)
            cep_radius_km = round(1.0 + (forecast_dist_km * 0.15), 1)

            estimated_height_m = int(pixel_count * 1.5 + 2000)
            tilt_ratio = (speed_kmh * 1000.0) / max(estimated_height_m, 1000.0)
            vis_status = "Verticale / Eretto" if tilt_ratio < 0.3 else ("Inclinato / Shear Moderato" if tilt_ratio < 0.8 else "Obliquo / High Shear")

            area_delta = pixel_count - prev_pixel_count
            if area_delta > 120:
                lifecycle = "Genesi (Growth)"
            elif area_delta < -120:
                lifecycle = "Dissipazione (Decay)"
            else:
                lifecycle = "Maturità (Mature)"

            echo_top_km = round(6.0 + (pixel_count / 350.0), 1)
            if echo_top_km > 14.5: echo_top_km = 14.5

            vil_val = round(pixel_count * 0.038, 1)
            qpe_val = round(pixel_count * 0.022, 1)
            intensity_flag = "Intensa" if pixel_count > 800 else "Moderata"
            flash_rate = int(pixel_count * 0.035) if intensity_flag == "Intensa" else int(pixel_count * 0.01)
            motion_flag = "Rischio Stazionario / Backbuilding" if speed_kmh < 12.0 else "Avvezione Standard"

            props = {
                "id": f"CELL_{idx+1}",
                "height": estimated_height_m,
                "intensity": intensity_flag,
                "speed_kmh": speed_kmh,
                "eta": eta_string,
                "cep_km": cep_radius_km,
                "vis": vis_status,
                "lifecycle": lifecycle,
                "echo_top": f"{echo_top_km} km",
                "vil": f"{vil_val} kg/m²",
                "qpe": f"{qpe_val} mm/h",
                "flash_rate": f"{flash_rate} stim/min",
                "motion": motion_flag
            }

            if current_cell["polygon"]:
                p_props = props.copy()
                p_props["type"] = "volumetric_cell"
                features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [current_cell["polygon"]]}, "properties": p_props})

            if len(path_coords) > 1:
                features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": path_coords}, "properties": {"id": f"PATH_{idx+1}", "type": "actual_path"}})

            f_props = props.copy()
            f_props["type"] = "forecast_path"
            features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": forecast_coords}, "properties": f_props})

            c_props = props.copy()
            c_props["type"] = "centroid"
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [c_lon, c_lat]}, "properties": c_props})

        geojson_output = {"type": "FeatureCollection", "features": features}

    except Exception as e:
        print(f"Errore tracker: {e}")
        geojson_output = {"type": "FeatureCollection", "features": []}

    with open("cells.geojson", "w", encoding='utf-8') as f:
        json.dump(geojson_output, f, indent=4)
        print("cells.geojson aggiornato con successo.")

if __name__ == "__main__":
    main()
                      
