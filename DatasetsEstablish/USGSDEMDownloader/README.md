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
  <h3 align="center">USGSDEMDownloader</h3>

  <p align="center">
    Automated USGS 3DEP 1-meter DEM Data Download Tool
    <br />
    <a href="https://github.com/isKunner/TheSotrageCapacityOfCheckDam">Kevin Chen</a>
  </p>
</div>

## Overview

**USGSDEMDownloader** is a Python toolkit designed to automate the discovery and download of USGS 3DEP (3D Elevation Program) 1-meter resolution DEM (Digital Elevation Model) data. It provides a complete workflow from indexing available datasets to matching them with target satellite imagery areas and generating downloadable HTML reports.

<p align="right">(<a href="#top">back to top</a>)</p>

## USGS_download_index

**Batch Download USGS 3DEP 1-meter DEM Project Index Files**

### Functionality

Automatically retrieves and downloads the download link index files for all USGS 3DEP 1-meter DEM projects.

*   **Get Project List**: Crawls the USGS server to obtain a list of all available 1-meter DEM projects (formatted as `STATE_ProjectName`).
*   **Download Link Files**: Batch downloads the `0_file_download_links.txt` file for each project, which contains all TIF data download links for that project.
*   **Progress Display**: Provides visual progress bars and statistical information during download.
*   **Resume Support**: Automatically skips already downloaded files, avoiding duplicate downloads.

### API Reference

| Function | Description |
|:---------|:------------|
| `get_all_projects()` | Retrieves a list of all USGS 3DEP 1-meter DEM projects from the server. |
| `download_link_file(project_name, output_dir)` | Downloads the link file for a specific project. |
| `usgs_down_index(output_dir, delay=1)` | Main function, batch downloads link files for all projects. |

### Parameters

| Parameter | Description |
|:----------|:------------|
| `output_dir` | Output directory path for storing downloaded index files. |
| `delay` | Request delay time in seconds (default: 1), used to avoid server overload. |

### Output

*   **Index Files**: A series of `.txt` files named after the projects (e.g., `CA_San Francisco_2021.txt`), each containing all download links for TIF files in that project.

<p align="right">(<a href="#top">back to top</a>)</p>

## USGS_download

**Match USGS DEM Download Links Based on Satellite Imagery Range**

### Functionality

Automatically matches and finds corresponding USGS 1-meter DEM data download links based on the geographic extent of Google satellite imagery TIF files.

*   **Geographic Range Analysis**: Reads the bounds of TIF files to obtain the geographic coordinates of the four corners.
*   **State Identification**: Uses the USA states Shapefile to determine which state each TIF file is located in.
*   **Coordinate Conversion**: Converts geographic coordinates to UTM coordinates and calculates the file name matching pattern (e.g., `x123y456`).
*   **Link Matching**: Finds corresponding DEM download links in the pre-built index dictionary.
*   **Error Handling**: Records detailed error information (unlocated states, unmatched links, etc.) in a JSON file.

### API Reference

| Function | Description |
|:---------|:------------|
| `download_single_file(url, download_dir)` | Downloads a single file. |
| `get_file_name_part(lon, lat)` | Converts latitude and longitude to UTM coordinates and generates the filename matching pattern. |
| `get_tif_bounds(google_file_path)` | Reads the geographic bounds of a TIF file. Returns format: `[lower_left, lower_right, upper_right, upper_left]`. |
| `usgs_load_file(google_remote_root_dir, usgs_dem_index_dir, usa_states_shp_path, usgs_dem_down_link)` | Main function, processes all TIF files and generates a download link mapping JSON. |

### Parameters

| Parameter | Description |
|:----------|:------------|
| `google_remote_root_dir` | Root directory path of Google satellite imagery TIF files, used to calculate target location ranges. |
| `usgs_dem_index_dir` | Directory path storing index files generated by `USGS_download_index.py`. |
| `usa_states_shp_path` | Path to the USA states Shapefile, used to determine the geographic state. |
| `usgs_dem_down_link` | Output JSON file path for storing the mapping between TIF files and DEM download links. |

### Output

*   **JSON File**: A structured JSON file containing:
    *   Group name (folder name)
    *   TIF filenames
    *   Matched USGS DEM download links
    *   Error information (if any)

### Important Notes

*   The tool assumes the spatial range of TIF files does not exceed 10,000 meters.
*   Requires the `utm` library for coordinate conversion.
*   Requires the USA states Shapefile for geographic location judgment.

<p align="right">(<a href="#top">back to top</a>)</p>

## USGS_generation_html

**Generate HTML Pages to Display USGS DEM Download Links**

### Functionality

Generates independent, visually appealing HTML files based on the JSON file generated by `USGS_download`, for easy browsing and downloading of DEM data.

*   **Group-based Generation**: Generates an independent HTML file for each group in the JSON file.
*   **Visual Interface**: Provides a clean, modern web interface displaying all download links.
*   **Link Direct Access**: All links are clickable and open in a new tab.
*   **Error Filtering**: Automatically filters out error entries and displays valid download links only.

### API Reference

| Function | Description |
|:---------|:------------|
| `is_error_entry(links)` | Checks if a link entry is an error record (keys starting with `__`). |
| `generate_html_for_usgs_down(json_path, output_dir)` | Main function, generates HTML files for all groups in the JSON. |

### Parameters

| Parameter | Description |
|:----------|:------------|
| `json_path` | Path to the input JSON file (generated by `USGS_download`). |
| `output_dir` | Output directory path for HTML files. |

### Output

*   **HTML Files**: Generates one HTML file per group (e.g., `GeoDAR_v11_dams_of_USA_group1_DEM_Links.html`), containing:
    *   Group name title
    *   List of all TIF files
    *   USGS DEM download links corresponding to each TIF file
    *   Clickable links for direct download

### HTML Template Features

*   Responsive design with modern styling
*   Clear hierarchy and visual grouping
*   Hover effects and interaction feedback
*   Automatic line wrapping for long links

<p align="right">(<a href="#top">back to top</a>)</p>

## USGS_down_process

**Process and Merge Downloaded USGS DEM Data**

### Functionality

Processes downloaded USGS DEM files by merging multiple source tiles and cropping them to match the geographic extent of reference satellite imagery.

*   **Batch Processing**: Automatically processes all groups and files defined in the JSON mapping file.
*   **Mosaic and Crop**: Merges multiple DEM source tiles and crops the result to match the exact extent of the reference imagery.
*   **Smart File Management**: Checks if all required source files are available before processing.
*   **Optional Cleanup**: Supports automatic deletion of processed source files to save disk space, with complete deletion records.

### API Reference

| Function | Description |
|:---------|:------------|
| `generation_usgs(google_remote_root_dir, usgs_dem_index_dir, dam_usgs_dem_down_file, usgs_dem_root_path, is_delete_file=False, usgs_dem_delete_info=None)` | Main function, processes DEM files by merging sources and cropping to reference extent. |

### Parameters

| Parameter | Description |
|:----------|:------------|
| `google_remote_root_dir` | Root directory containing Google satellite imagery TIF files, used as reference for cropping extent. |
| `usgs_dem_index_dir` | Path to the JSON file generated by `USGS_download.py`, containing the mapping between TIF files and DEM download links. |
| `dam_usgs_dem_down_file` | Directory path where downloaded USGS DEM files are stored. |
| `usgs_dem_root_path` | Output directory path for storing processed (cropped and merged) DEM files. |
| `is_delete_file` | Boolean flag indicating whether to delete source DEM files after successful processing (default: `False`). |
| `usgs_dem_delete_info` | Path to CSV file for recording deletion information. Required if `is_delete_file` is `True`. |

### Output

*   **Processed DEM Files**: Cropped and merged DEM TIF files organized by group, matching the geographic extent of the reference satellite imagery.
*   **Deletion Records** (optional): CSV file containing deletion history with timestamps, deleted filenames, and original URLs.

### Important Notes

*   This module depends on `DEMAndRemoteSensingUtils` for geospatial processing functions (`merge_sources_to_reference`, etc.).
*   Ensure all required source DEM files are downloaded before processing.
*   Use the deletion feature with caution as deleted files cannot be easily recovered.

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage Example

```python
from USGSDEMDownloader import usgs_down_index, usgs_load_file, generate_html_for_usgs_down, generation_usgs

# Step 1: Download USGS DEM index files
usgs_down_index("./usgs_index", delay=1)

# Step 2: Match DEM links based on satellite imagery
usgs_load_file(
    google_remote_root_dir="./google_images",
    usgs_dem_index_dir="./usgs_index",
    usa_states_shp_path="./USA_States.shp",
    usgs_dem_down_link="./download_links.json"
)

# Step 3: Generate HTML download pages
generate_html_for_usgs_down("./download_links.json", "./html_output")

# Step 4: Down DEM files (merge and crop)
Downloading files through plugins, such as 'DownThemALL!', is more complete and suitable for a large number of file downloads

# Step 5: Process downloaded DEM files (merge and crop)
generation_usgs(
    google_remote_root_dir="./google_images",
    usgs_dem_index_dir="./download_links.json",
    dam_usgs_dem_down_file="./downloaded_dem",
    usgs_dem_root_path="./processed_dem",
    is_delete_file=True,
    usgs_dem_delete_info="./deletion_records.csv"
)
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Requirements

*   Python 3.x
*   requests
*   utm
*   geopandas
*   rasterio
*   shapely
*   tqdm
*   pandas

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
