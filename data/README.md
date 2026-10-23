Analysis-ready data for the wildfire severity manuscript.


## EPA Level III Ecoregions

EPA Level III ecoregion boundaries were obtained from the U.S. Environmental Protection Agency (EPA) and used for spatial analysis of wildfire burn severity across the western United States.

**Data source:** U.S. Environmental Protection Agency (EPA)

**Download:** https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states

**Dataset:** Level III Ecoregions of the Conterminous United States (`us_eco_l3`).

To reproduce the spatial analyses:

1. Visit the EPA download page linked above.
2. Expand the **Level III Ecoregions** section.
3. Download the **US Level III Ecoregions shapefile** (without state boundaries).
4. Extract the downloaded ZIP file.
5. Keep the `.shp`, `.shx`, `.dbf`, and `.prj` files together in the same directory.

The Python scripts reference the shapefile as `us_eco_l3.shp`. Update the local file path in the scripts as needed.
