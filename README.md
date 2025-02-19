# **Nitrogen Flow Model for Chesapeake Bay Region**

## **Overview**
This repository contains Python scripts for modeling nitrogen flows in agricultural systems, focusing on the Chesapeake Bay region. The model integrates crop and animal production stages, tracking nitrogen use, losses, and recycling across multiple stages of the agricultural supply chain. 

The data used in this model is sourced from the **United States Department of Agriculture (USDA)** and is available on their official website.

## **Key Components**
This model builds upon previous versions by incorporating a methodology to connect crop and animal production. Additionally, it removes outliers in average yield data to improve accuracy. The complementary analysis for this version is located in `codes/crop-animal_transition_methodology2.ipynb`.

### **Nitrogen Flow Stages**
The model includes three progressive levels of analysis:

1. **Crop Stages (`trade_simulation5`)**  
   - Analyzes nitrogen inputs, crop yields, fertilizer application, and nitrogen loss in crop production.

2. **Crop and Live Animal Stages (`trade_simulation_V2_5`)**  
   - Incorporates nitrogen flows in livestock production, including feed conversion ratios and manure recycling.

3. **Crop, Live Animal, and Animal Product Stages (`trade_simulation_V3_5`)**  
   - Extends the analysis to include meat and dairy processing, tracking nitrogen recycling and losses in animal products.

### **Key Variables Tracked**
- **Fertilizer nitrogen input**
- **Crop nitrogen uptake**
- **Animal feed nitrogen intake**
- **Manure nitrogen recycling**
- **Nitrogen losses at different stages (crop production, animal rearing, food processing, and human consumption)**

## **Datasets**
- **Planting Area Data** (`harvested_area_county_filled.csv`): Contains harvested area data for different crops.
- **Live Animal Inventory** (`inventory_animals_county_filled.csv`): Provides livestock headcounts at the county level.
- **Agricultural Production** (`ag_production_filled.csv`): Reports crop production values in bushels and tons.
- **Trade Data** (`sctg*_trade_disaggregated.csv`): Contains trade flows for various agricultural commodities, including crops, livestock, and meat products.

## **Model Outputs**
The model generates several output files stored in `output/results_2017/nitrogen_flow_model_v4/`:

- `crop_interlayer_flows.csv`: Captures nitrogen flows within crop production stages.
- `crop_processing_nitrogen.csv`: Provides processed nitrogen values for crops.
- `animal_stage_nitrogen.csv`: Details nitrogen flows within live animal stages.
- `meat_interlayer_flows.csv`: Tracks nitrogen in processed animal products.
- `crop_intralayer_flows.csv`: Represents nitrogen movement in crop trading networks.
- `animal_intralayer_flows.csv`: Represents nitrogen movement in live animal trading networks.
- `meat_intralayer_flows.csv`: Represents nitrogen movement in meat trading networks.

## **Chesapeake Bay Focus**
This model is designed to narrow down nitrogen loss estimates specifically within the **Chesapeake Bay region**, where excessive nitrogen runoff contributes to water quality degradation. By quantifying nitrogen losses at different stages, the model helps identify intervention points for reducing nutrient pollution in the watershed.

## **How to Run the Model**
1. Ensure all required datasets are placed in the appropriate directory.
2. Modify `os.chdir("path/to/model_and_scenarios/scenarios")` to reflect the correct local file path.
3. Run the Python script (`nitrogen_flow_model.py`) in a Jupyter Notebook or command-line interface.

## **Acknowledgments**
This model is based on USDA data sources and trade flow estimates derived from publicly available datasets.
