<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div>
  <h3 align="center">DatasetsEstablish</h3>

  <p align="center">
    Various utility modules used in processing Digital Elevation Models (DEMs) and Remote Sensing
    <br />
    <a href="https://github.com/isKunner/TheSotrageCapacityOfCheckDam">Kevin Chen</a>
  </p>
</div>

## create_silted_land_label_from_shp

**Generate Label Annotation Files from Shapefile (for Silted Land Parcels)**

*   **Background**:
    1.  The Shapefile consists of multiple Polygons, each representing a silted land parcel (formed by a check dam).
    2.  Corresponding Google satellite imagery TIF data has been clipped based on the Shapefile, where each polygon was expanded by a certain number of pixels and then cropped to 1024*1024 pixels.
    3.  The goal is to convert these Polygons into LabelMe annotations.
*   **Steps**:
    1.  **Read Source Data**: Load the Shapefile containing the feature boundary information.
    2.  **Iterate Processing**: Loop through each feature (parcel) in the Shapefile.
    3.  **Match Imagery**: Find the corresponding satellite image (.tif) based on the feature ID.
    4.  **Read Image Metadata**: Obtain image dimensions, coordinate system (CRS), and geotransform information.
    5.  **Coordinate Alignment**: Perform projection transformation if the Shapefile and image have different CRS.
    6.  **Coordinate Transformation**: Use the image's transformation matrix to accurately convert the feature's geographic coordinates to pixel coordinates on the image.
    7.  **Handle Geometry**: Properly process Polygon and MultiPolygon types.
    8.  **Read/Merge Labels**: If a LabelMe annotation file already exists for the image, load it and append the newly converted shape, avoiding overwriting existing labels.
    9.  **Deduplication**: Check if the newly added shape already exists in the old label to prevent duplicates.
    10. **Build JSON**: Integrate all shapes and metadata according to the LabelMe format requirements.
    11. **Output File**: Save the final JSON data as a .json file.
    12. **Logging & Error Tracking**: Use `logging` throughout the process to record detailed information to a file, and categorize specific errors into a CSV file for easy debugging and auditing.

<p align="right">(<a href="#top">back to top</a>)</p>

## generate_silted_land_shp_from_label

Unify Slope and Road LabelMe Annotations into a Single Shapefile (with Deduplication and Metric Calculation)

### Functionality

Aggregates "slope" and "road" labeled polygons from multiple分散 LabelMe JSON files into two separate, deduplicated ESRI Shapefiles. The script also calculates precise geospatial dimensions (length, width) and orientation angle for each object.

*   **Aggregate:** Iterates through LabelMe JSON files in a specified directory, extracting polygons with target labels ("slope", "road").
*   **Georeference:** Uses corresponding GeoTIFF images to transform polygon vertices from image pixel coordinates to geographic coordinates.
*   **Deduplicate:** Identifies and removes near-duplicate polygons (within the same or across different files) based on Intersection over Union (IoU).
*   **Unify:** Saves the resulting unique polygons for each label type into separate Shapefile outputs.
*   **Metric Calculation:** Calculates precise metric dimensions (Length in meters, Width in meters) and the main orientation Angle (degrees) for each retained polygon using its Minimum Area Bounding Rectangle (MABR).

### Background

There exists a large number of manually annotated LabelMe JSON files containing "slope" and "road" labels. These annotations are scattered across individual files. The goal is to consolidate, clean, and produce these annotations into a standard GIS data format suitable for further analysis.

### Inputs

1.  **LabelMe JSON Directory (`--json-label-dir`)**: Path to the folder containing the `.json` annotation files.
2.  **GeoTIFF Image Directory (`--tif-dir`)**: Path to the folder containing the corresponding `.tif` satellite/aerial images. The `imagePath` field in the JSON is used to match the correct TIF file.
3.  **Target Labels (`--target-labels`)**: A list of labels to process (default: `["slope", "road"]`).
4.  **Overlap Threshold (`--overlap-threshold`)**: IoU threshold (0.0 to 1.0) above which polygons are considered duplicates (default: `0.5`).
5.  **Processing Mode (`--processing-mode`)**: Either `single` (uses original TIF geoinfo) or `individual` (recalculates MABR for each shape individually) (default: `single`).
6.  **Output Paths (`--output-shp-path`, `--log-file-path`)**: Desired paths for the output Shapefile(s) and log file.

### Outputs

1.  **Shapefiles:**
    *   One Shapefile for `slope` polygons (e.g., `merged_slope_deduplicated.shp`).
    *   One Shapefile for `road` polygons (e.g., `merged_road_deduplicated.shp`).
    *   Each Shapefile contains the following attributes for every feature:
        *   `geometry`: The unified Polygon geometry in geographical coordinates.
        *   `label`: The original label from the JSON (e.g., "slope").
        *   `angle_deg`: The orientation angle (degrees) of the minimum bounding rectangle, defined as the clockwise angle from the X-axis to the first side touched during clockwise rotation. Range (0, 90].
        *   `length_m`: The length (meters) of the minimum bounding rectangle, corresponding to the side perpendicular to the `width_m` side.
        *   `width_m`: The width (meters) of the minimum bounding rectangle, corresponding to the side defined by the `angle_deg`.
        *   `filename`: The name of the source TIF file.
2.  **Log File:** A text log detailing the processing steps, files read, polygons found, duplicates removed, and any warnings/errors encountered.

### Methodology

1.  **Iteration & Matching:** The script scans the JSON directory. For each JSON file, it identifies the corresponding TIF file.
2.  **Geoinformation Retrieval:** Geotransform matrix and Coordinate Reference System (CRS) are extracted from the TIF file.
3.  **Parsing & Filtering:** Polygons from the JSON are parsed. Only those matching the `--target-labels` and of type 'polygon' are selected.
4.  **Pixel-to-Geo Transformation:** Polygon vertices are converted from pixel coordinates to geographical coordinates using the TIF's geotransform. This geo-referenced polygon is the definitive geometry used.
5.  **Accurate Metric Calculation:**
    *   If the TIF's CRS is geographic (e.g., WGS84), the polygon is temporarily reprojected to a local UTM zone for accurate calculations.
    *   `cv2.minAreaRect` is applied to the (potentially projected) polygon coordinates to find the minimum area bounding rectangle.
    *   The angle, width, and height are extracted from `minAreaRect`. For OpenCV >= 4.5, the angle range is (0, 90] degrees, clockwise from the X-axis to the first touched side. The corresponding dimension is assigned as `width_m`.
6.  **Deduplication (Union-Find):**
    *   All valid geo-polygons are collected.
    *   An overlap matrix is computed using IoU.
    *   A Union-Find data structure groups overlapping polygons (based on `--overlap-threshold`).
    *   From each group, one representative polygon is selected (currently the first encountered).
7.  **Output Generation:** Separate GeoDataFrames are created for each label type using the representative polygons and their calculated metrics. These are saved as Shapefiles.

## get_google_for_spatial_info.py

**Generate 1024x1024 Remote Sensing Images from Shapefile and Clip Corresponding Copernicus DEM Data**

*   **Important Notes**:
    *   To export remote sensing images of exactly 1024x1024 pixels, the ArcGIS Pro window (canvas/layout) must also be set to 1024x1024 pixels. Otherwise, the exported size will be uncontrollable due to differing scales.
    *   **Specific Operations**:
        1.  Create a Layout, as the Map window cannot precisely adjust scale.
        2.  Set the layout size to 1:1.
        3.  Run the corresponding ArcGIS code within this layout interface.
        4.  ❗ This script **can only run within the ArcGIS environment**.
        5.  ❗ ❗ ❗ Additionally, **right-click the layout and activate the map**.
    *   The resolution of the generated DEM differs from the remote sensing image and therefore requires resampling. Code from `crop_dem_from_dem.py` or `resize_dem.py` can be used.

<p align="right">(<a href="#top">back to top</a>)</p>

## static_info_from_check_dam.py

**Statistically Analyze the Number of Check Dams by Prefecture-level City from a Shapefile and Perform Stratified Sampling**

| Function Name                                 | Functionality                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:----------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `get_shp_file_info(shp_file_path)`            | Prints and returns basic information about the specified Shapefile (path, record count, geometry type, coordinate system, fields, bounding extent).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `convert_geojson_2_shp()`                     | *(Auxiliary Function)* Splits TianDiTu downloaded administrative division data in GeoJSON format by geometry type and converts it to Shapefile.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `check_dam_correct(check_dam_file)`           | Checks the uniqueness of OBJECTIDs in the check dam Shapefile and modifies duplicate OBJECTIDs to unique values.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `statistical_check_dams(...)`                 | Counts the number of check dams within each county-level administrative division. Achieved via spatial join, handling duplicate features located on boundaries. Outputs a CSV file containing Province, City, County names, and dam counts.                                                                                                                                                                                                                                                                                                                                                                                                |
| `get_dams_by_admin_level_with_hierarchy(...)` | **Core Sampling Function**. Performs stratified sampling of check dams based on a specified administrative level (e.g., county):<br>• Spatially joins administrative divisions with check dams<br>• Aggregates all OBJECTIDs within each administrative region<br>• Performs random sampling based on a set ratio (`sample_ratio`)<br>• **Outputs**:<br>&nbsp;&nbsp;• A CSV file of sampling results (including region name, total count, sampled count)<br>&nbsp;&nbsp;• A Shapefile containing all sampled dams<br>&nbsp;&nbsp;• A detailed JSON file (containing metadata, sampling details per region, lists of sampled/remaining IDs) |

*   **Example Code Snippet**:

    ```python
    province_file = r"C:\Users\Kevin\Documents\ResearchData\AdministrativeDivision\ProvincialBoundary.shp"
    city_file = r"C:\Users\Kevin\Documents\ResearchData\AdministrativeDivision\CityBoundary.shp"
    county_file = r"C:\Users\Kevin\Documents\ResearchData\AdministrativeDivision\CountryBoundary.shp"
    check_dam_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\check_dam_dataset.shp"
    csv_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\sampled_dams.csv"
    selected_shp_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dam_dataset.shp"
    selected_csv_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dams.csv"

    # Get intuitive information about the shp file
    get_shp_file_info(check_dam_file)

    # Statistically analyze the geographical distribution of the shp file
    # statistical_check_dams(province_file=province_file, city_file=city_file, county_file=county_file, check_dam_file=check_dam_file, csv_file=csv_file)

    # Dataset clipping: 8:2 ratio resulted in 10006 samples
    # get_dams_by_admin_level_with_hierarchy(admin_file=county_file, check_dam_file=check_dam_file, csv_file=selected_csv_file, shp_file=selected_shp_file, province_file=province_file, city_file=city_file, admin_level='County')

    # Dataset sampling: Too many samples, so reduce by half based on the above result, finally obtaining 4991 samples
    selected_shp_file_5000 = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dam_dataset_5000.shp"
    selected_csv_file_5000 = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dams_5000.csv"
    # get_dams_by_admin_level_with_hierarchy(admin_file=county_file, check_dam_file=selected_shp_file,
    #                                        csv_file=selected_csv_file_5000, shp_file=selected_shp_file_5000,
    #                                        province_file=province_file, city_file=city_file, admin_level='County', sample_ratio=0.5)
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

