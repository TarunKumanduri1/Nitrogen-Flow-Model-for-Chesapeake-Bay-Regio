'''
Difference with previous version of the model is that here we implement a
methodology to connect crop and animal production. A complementary analysis for
this version of the mode is located at
codes/crop-animal_transition_methodology2.ipynb.
Further we remove outliers in average yield data from analysis.
'''

import pandas as pd
import numpy as np
import os

user = 'G22824390'
os.chdir("C:\\Users\\%s\\Box\\AT Research\\AG\\Model\\codeshare\\codeshare\\model_and_scenarios\\scenarios"%user)
#os.chdir("path/to/model_and_scenarios/scenarios") # CHANGE TO PATH IN LOCAL MACHINE

folder_name_output = 'initial'
folder_name_plots = 'initial'
# Import data
## Planting area
area = pd.read_csv('../../output/data_model_2017/harvested_area_county_filled.csv')
## Live animals inventory
inventory = pd.read_csv('../../output/data_model_2017/inventory_animals_county_filled.csv')
## Agricultural production (USDA)
agprod = pd.read_csv('../../output/trade_disaggregation_2017/ag_production_filled.csv', index_col=0)
## Trade data
ag_trade1 = pd.read_csv('../../output/trade_disaggregation_v2/sctg1-2-3_trade_disaggregated.csv', index_col=0)
ag_trade2 = pd.read_csv('../../output/trade_disaggregation_v2/sctg4_trade_disaggregated.csv', index_col=0)
ag_trade = pd.concat([ag_trade1, ag_trade2])
egg_trade = ag_trade[ag_trade['commodity']=='egg'].copy()
ag_trade = ag_trade[~(ag_trade['commodity']=='egg')]
ag_trade['commodity'] = ag_trade['commodity'].replace({'corn':'corn_grain', 'wheat':'wheat_grain',
                                                      'hay':'alfalfa_hay'})
meat_trade1 = pd.read_csv('../../output/trade_disaggregation_v2/sctg5_trade_disaggregated.csv', index_col=0)
meat_trade2 = pd.read_csv('../../output/trade_disaggregation_v2/sctg7_trade_disaggregated.csv', index_col=0)
meat_trade = pd.concat([meat_trade1, meat_trade2, egg_trade])
meat_trade['commodity'] = meat_trade['commodity'].replace({'chicken_meat':'broiler', 
                        'beef':'cow_beef', 'pork':'pig', 'milk':'cow_milk', 'egg':'layer'})

# Convert planting area dataframe from wide to long format
cols = area.columns
area_df = pd.melt(area, id_vars=cols[:4], value_vars=cols[4:])
area_df = area_df.rename(index=str, columns={'variable':'commodity', 'value':'harvested_area_acre'})
area_df['harvested_area_acre'] = area_df['harvested_area_acre'].replace(0, np.nan)
# Convert live animal inventory dataframe from wide to long format
cols = inventory.columns
inventory_df = pd.melt(inventory, id_vars=cols[:4], value_vars=cols[4:])
inventory_df = inventory_df.rename(index=str, columns={'variable':'commodity', 'value':'inventory_head'})
inventory_df['inventory_head'] = inventory_df['inventory_head'].replace(0, np.nan)
# Convert agricultural production from wide to long format
agprod = agprod.rename(index=str, columns={'corn_production_bushel_filled':'corn_grain', 
        'cornsilage_production_tons_filled': 'corn_silage', 
        'hay_production_tons_filled':'alfalfa_hay', 'otherhay_production_tons_filled':'other_hay',
       'soybean_production_bushel_filled':'soybean', 'wheat_production_bushel_filled':'wheat_grain',
       'broiler_sales_head_filled':'broiler', 'layer_inventory_head_filled':'layer',
       'cowbeef_inventory_head_filled':'cow_beef', 'cowmilk_inventory_head_filled':'cow_milk',
       'pig_inventory_head_filled':'pig'})
cols = agprod.columns
agprod_df = pd.melt(agprod, id_vars=['FIPS', 'FIPS_ST', 'state'], value_vars=cols[4:-1])
agprod_df = agprod_df.rename(index=str, columns={'variable':'commodity', 'value':'production_bushel&tons'})

# Merge production and planting area data
dat = area_df.merge(agprod_df, how='outer', on=['FIPS', 'commodity', 'state'])

'''
GLOSSARY
PA: harvested_area (ha)
PQ: production_kg (kg)
AY: average_yield (kg/ha)
YNc: yield_nitrogen_content (%)
YN: yield_nitrogen (kg N/ha)
RCr: residue_ratio (ratio)
RNc: residue_nitrogen_content (kg N/kg residue)
DM: dry_matter (ratio)
CRN: residue_nitrogen (kg N/ha)
ADCN: domestic_crop_nitrogen_area (kg N/ha)
DCN: domestic_crop_nitrogen (kg N)
FNAr: fertilizer_nitrogen_application_rate (kg N/kg crop)
FNAR: fertilizer_nitrogen_application_area (kg N/ha)
FN: fertilizer_nitrogen (kg N)
BNFr: biological_nitrogen_fixation_rate (kg N/kg crop)
BNFR: biological_nitrogen_fixation_area (kg N/ha)
BNF: biological_nitrogen_fixation (kg N)
TNN: total_new_input_nitrogen (kg N)
CPr: crop_processing_ratio (ratio)
PDCN: domestic_crop_processing_nitrogen (kg N)
WN2: nitrogen_waste2 (kg N) - crop waste N (from processing)
RR1: nitrogen_recycling_rate1 (%)
R1: nitrogen_recycling1 (kg N) - crop waste N (from processing) that will be recycled back
LN2: nitrogen_loss2 (kg N) - nitrogen loss from the processing of crops
ImCN-CG: import_crop_processing_nitrogen (kg N)
ExCN-CG: export_crop_processing_nitrogen (kg N)
PCN: processed_crop_nitrogen (kg N)
DCLAr-Pb: diet_composition_broiler_ratio
DCLAr-Pl: diet_composition_layer_ratio
DCLAr-P: diet_composition_pig_ratio
DCLAr-Cm: diet_composition_cowmilk_ratio
DCLAr-Cb: diet_composition_cowbeef_ratio
DCLANc-Pb nitrogen_content_diet_broiler (%)
DCLANc-Pl nitrogen_content_diet_layer (%)
DCLANc-P nitrogen_content_diet_pig (%)
DCLANc-Cm nitrogen_content_diet_cowmilk (%)
DCLANc-Cb nitrogen_content_diet_cowbeef (%)
LANc: animal_nitrogen_content (%)
FCR: feed_conversion_ratio
DCLANc: nitrogen_content_diet (%)
LANU: animal_nitrogen_uptake (%)
DLAP: inventory_head (head)
LAWg: animal_weight_gain (kg/head)
DLAN: domestic_animal_nitrogen (kg N)
ExLAN: export_animal_nitrogen (kg N)
ImLAN: import_animal_nitrogen (kg N)
LAN : animal_nitrogen (kg N)
FCN: feed_nitrogen_crop (kg N)
DCLAN-(Pl, Pb, P, CB, Cm)-CG: crop_diet_percentage_corn (%)
DCLAN-(Pl, Pb, P, CB, Cm)-WG: crop_diet_percentage_wheat (%)
DCLAN-(Pl, Pb, P, CB, Cm)-Sb: crop_diet_percentage_soybean (%)
DCLAN-(Pl, Pb, P, CB, Cm)-CS: crop_diet_percentage_cornsilage (%)
DCLAN-(Pl, Pb, P, CB, Cm)-AlH: crop_diet_percentage_alfalfa (%)
DCLAN-(Pl, Pb, P, CB, Cm)-OtH: crop_diet_percentage_otherhay (%)
DCLANW-(Pl, Pb, P, CB, Cm)-CG: nitrogen_content_diet_weight_corn (%)
DCLANW-(Pl, Pb, P, CB, Cm)-WG: nitrogen_content_diet_weight_wheat (%)
DCLANW-(Pl, Pb, P, CB, Cm)-Sb: nitrogen_content_diet_weight_soybean (%)
DCLANW-(Pl, Pb, P, CB, Cm)-CS: nitrogen_content_diet_weight_cornsilage (%)
DCLANW-(Pl, Pb, P, CB, Cm)-AlH: nitrogen_content_diet_weight_alfalfa (%)
DCLANW-(Pl, Pb, P, CB, Cm)-OtH: nitrogen_content_diet_weight_otherhay (%)
LASr: animal_slaughtering_rate
ACN: animal_carcass_milked_laid_nitrogen (kg N)
RR3: nitrogen_recycling_rate3 (kg N)
R3: nitrogen_recycling3 (kg N) - animal N waste (from slaughtering, milking and laying) to be recycled back as feed
LN4: nitrogen_loss4 (kg N) - nitrogen loss from animal slaughtering, milking and laying
APPr: animal_product_processing_rate
DAPN: domestic_animal_product_nitrogen (kg N)
RR4: nitrogen_recycling_rate4
R4: nitrogen_recycling4 (kg N) - processed animal product N waste  to be recycled back as feed
LN5: nitrogen_loss5 (kg N) - nitrogen loss from animal byproducts processing
ExAPN: export_meat_nitrogen (kg N)
ImAPN: import_meat_nitrogen (kg N)
APN: animal_product_nitrogen (kg N)
APCr: animal_product_consumption_rate
APCN: animal_product_nitrogen_consumption (kg N)
LN6: nitrogen_loss6 (kg N) - food N waste
LN7: nitrogen_loss7 (kg N) - human N waste
FR3N: feed_nitrogen_recycled_slaughtering_waste (kg N)
FR4N: feed_nitrogen_recycled_meat_waste (kg N)
TFN: total_feed_nitrogen (kg N)
WN3: manure_feed_nitrogen_waste (kg N)
RR2: nitrogen_recycling_rate2
R2: nitrogen_recycling2 (kg N) - feed and manure N (from crop and manure processing)
LN3: nitrogen_loss3 (kg N) - nitrogen loss from feed and manure waste
RNr: recycled_nitrogen_crop_share
RN1:nitrogen_recycling1_from_crop (kg N)
RN2: nitrogen_recycling1_from_manure (kg N)
TRN: total_recycled_nitrogen_input (kg N)
TN: total_nitrogen_input (kg N)
LN1: nitrogen_loss1 (kg N) - nitrogen loss from crop planting
'''

# Nitrogen flow model
dat = dat[dat['commodity'].isin(['corn_grain', 'wheat_grain', 'soybean', 'alfalfa_hay',
       'corn_silage', 'other_hay'])]
## Convert planting area from ac to ha
dat['harvested_area'] = dat['harvested_area_acre'] * 0.404686
## Convert production from bushels and tons to kg
dict_conv_factors_production = {'corn_grain':56/2.2, 'wheat_grain':60/2.2, 'soybean':60/2.2, 
                                'alfalfa_hay':1000, 'corn_silage':1000, 
                                    'other_hay':1000}
dat['conv_factor_production_kg'] = dat['commodity'].map(dict_conv_factors_production)
dat['production_kg'] = dat['production_bushel&tons'] * dat['conv_factor_production_kg']
#Calculate average yield (production unit/acre)
dat['average_yield_raw'] = dat['production_bushel&tons'] / dat['harvested_area_acre']

## Calculate average yield (kg/ha)
dat['average_yield'] = dat['production_kg'] / dat['harvested_area']

##############################################################################
crop_list = ['alfalfa_hay', 'corn_grain', 'corn_silage', 'other_hay', 'soybean', 'wheat_grain']
for c in crop_list:
    aux = dat[dat['commodity']==c].copy()
    lower, upper = (aux['average_yield'].quantile(0.005), aux['average_yield'].quantile(0.995))
    aux2 = aux[(aux['average_yield']>upper)|(aux['average_yield']<lower)].copy()
    aux2['bool_outlier_%s'%c] = 1
    dat = dat.merge(aux2[['FIPS', 'commodity', 'bool_outlier_%s'%c]], how='left', 
                                        on=['FIPS', 'commodity'])

cols = [i for i in dat.columns if 'bool_outlier' in i]
dat['bool_outlier'] = dat[cols].sum(axis=1)
## Remove outliers from crop dataframe
dat = dat[dat['bool_outlier']!=1].copy()
cols = [i for i in dat.columns if 'bool_outlier' in i]
dat = dat.drop(columns=cols)

##############################################################################

## Calculate Yield Nitrogen, wet (kg N/ha)
# {'corn_grain':, 'wheat_grain':, 'soybean':, 
#  'alfalfa_hay':, 'corn_silage':, 'other_hay':}
dict_yield_N_content = {'corn_grain':1.2, 'wheat_grain':2.3, 'soybean':4.8, 
                        'alfalfa_hay':2.59, 'corn_silage':0.37, 'other_hay':1.65}
dat['yield_nitrogen_content'] = dat['commodity'].map(dict_yield_N_content)
dat['yield_nitrogen'] = dat['average_yield'] * dat['yield_nitrogen_content']/100

## Calculate Residue Nitrogen, dry (kg N/ha)
dict_residue_ratio = {'corn_grain':0.845, 'wheat_grain':1.32, 'soybean':0.87, 
                      'alfalfa_hay':0, 'corn_silage':0, 'other_hay':0}
dict_residue_nitrogen_content = {'corn_grain':0.8, 'wheat_grain':0.75, 'soybean':2, 
                           'alfalfa_hay':0, 'corn_silage':0, 'other_hay':0}
dict_dry_matter = {'corn_grain':84.5, 'wheat_grain':88, 'soybean':87, 
                   'alfalfa_hay':87, 'corn_silage':35, 'other_hay':87}
dat['residue_ratio'] = dat['commodity'].map(dict_residue_ratio)
dat['residue_nitrogen_content'] = dat['commodity'].map(dict_residue_nitrogen_content)
dat['dry_matter'] = dat['commodity'].map(dict_dry_matter)
dat['residue_nitrogen'] = dat['average_yield'] * dat['residue_ratio'] * dat['residue_nitrogen_content'] / 100

## Calculate Domestic crop N (whole plant N) per area for each crop type (kg N/ha)
dat['domestic_crop_nitrogen_area'] = dat['yield_nitrogen'] + dat['residue_nitrogen']

## Calculate Domestic crop N (whole plant N) weight for each crop type (kg N)
dat['domestic_crop_nitrogen'] = dat['harvested_area'] * dat['domestic_crop_nitrogen_area']

## Calculate New input N added to each crop type (kg N/ha)
### Calculate Fertilizer Nitrogen Application Rate (FNAR) per Area for each crop type (kg N/ha)
''' 
fertilizer N application rate is in unit lb N/unit of expected yield. Unit 
of expected yield is different for each crop, it's defined as (unit production is
provided in census)/(acre). So we need to multiply by yield data before conversion
 to kg/ha. Then, a conversion factor is applied to get values in kg N/ha.
'''
dict_fertilizer_nitrogen_appl_rate = {'corn_grain':1, 'wheat_grain':1.25, 'soybean':0, 
                                      'alfalfa_hay':0, 'corn_silage':7, 'other_hay':50}
conv_factor = 1.121 # (lb/acre) to (kg/ha)
dat['fertilizer_nitrogen_application_rate'] = dat['commodity'].map(dict_fertilizer_nitrogen_appl_rate)
dat['fertilizer_nitrogen_application_area'] = dat['average_yield_raw'] * dat['fertilizer_nitrogen_application_rate'] * conv_factor
### Calculate Fertilizer Nitrogen (FN) for each crop type (kg N)
dat['fertilizer_nitrogen'] = dat['harvested_area'] * dat['fertilizer_nitrogen_application_area']
### Calculate Biological Nitrogen Fixation Rate (BNFR) weight for each crop type (kg N)
dict_biological_nitrogen_fixation_rate = {'corn_grain':0, 'wheat_grain':0, 'soybean':5.3, 
                                          'alfalfa_hay':75, 'corn_silage':0, 'other_hay':0}
dat['biological_nitrogen_fixation_rate'] = dat['commodity'].map(dict_biological_nitrogen_fixation_rate)
dat['biological_nitrogen_fixation_area'] = dat['average_yield_raw'] * dat['biological_nitrogen_fixation_rate'] * conv_factor
### Calculate Biological Nitrogen Fixation (BNF) for each crop type (kg N)
dat['biological_nitrogen_fixation'] = dat.apply(lambda x: x['domestic_crop_nitrogen']
    if x['commodity'] in ['soybean', 'alfalfa_hay'] else x['harvested_area'] * x['biological_nitrogen_fixation_area'], axis=1)
#dat['harvested_area'] * dat['biological_nitrogen_fixation_area']
### Calculate Total New input N (TNN) added to each crop type from both fertilizer and BNF (kg N)
dat['total_new_input_nitrogen'] = dat['fertilizer_nitrogen'] + dat['biological_nitrogen_fixation']

## Calculate Crop Processing ratio for each crop type (ratio)
dat['crop_processing_ratio'] = dat['yield_nitrogen'] / dat['domestic_crop_nitrogen_area']

## Calculate Domestic Procesed Crop N weight for each crop type (kg N) = yield N, Wet (kg N)
dat['domestic_crop_processing_nitrogen'] = dat['domestic_crop_nitrogen'] * dat['crop_processing_ratio']

## Calculate Crop Processing N Waste (WN2) for each crop type (kg N) = Residue N, Dry (kg N)
dat['nitrogen_waste2'] = dat['domestic_crop_nitrogen'] - dat['domestic_crop_processing_nitrogen']

## Calculate N recycled back from N waste of crop processing (Crop Residue) for each crop type (kg N)
dict_nitrogen_recycling_rate1 = {'corn_grain':0.35, 'wheat_grain':0.35, 'soybean':0.35, 
                                 'alfalfa_hay':0, 'corn_silage':0, 'other_hay':0}
dat['nitrogen_recycling_rate1'] = dat['commodity'].map(dict_nitrogen_recycling_rate1)
dat['nitrogen_recycling1'] = dat['nitrogen_waste2'] * dat['nitrogen_recycling_rate1']
R1_total = dat[['FIPS', 'nitrogen_recycling1']].groupby('FIPS').sum().reset_index().rename(
    index=str, columns={'nitrogen_recycling1':'nitrogen_recycling1_total'})

## Calculate N loss in Stage 2: Parts of Crop Processing N Waste lost to the environment for each crop type (kg N)
dat['nitrogen_loss2'] = dat['nitrogen_waste2'] - dat['nitrogen_recycling1']

## Calculate Nitrogen embedded in imports and exports of crops
### Calculate proportion of imports and exports for each link
ag_trade_crop = ag_trade[ag_trade['commodity'].isin(['corn_grain', 'wheat_grain', 'soybean', 
                                                     'alfalfa_hay', 'corn_silage', 'other_hay'])].copy()
ag_trade_crop_export = ag_trade_crop.groupby(['Source', 'commodity']).agg({
    'flow_kg_commodity':np.nansum}).reset_index().rename(index=str, columns={'flow_kg_commodity':'export_kg'})
ag_trade_crop2 = ag_trade_crop.merge(ag_trade_crop_export, how='left', on=['Source', 'commodity'])
ag_trade_crop2['export_proportion'] = ag_trade_crop2['flow_kg_commodity'] / ag_trade_crop2['export_kg']
ag_trade_crop2.dropna(subset=['export_proportion'], inplace=True)
### Add domestic crop processing Nitrogen data to trade data
cols = ['FIPS', 'commodity', 'domestic_crop_processing_nitrogen']
ag_trade_crop2 = ag_trade_crop2.merge(dat[cols], how='left', left_on=[
    'Source', 'commodity'], right_on=['FIPS', 'commodity'])
ag_trade_crop2['trade_crop_processing_nitrogen'] = ag_trade_crop2['domestic_crop_processing_nitrogen'] \
    * ag_trade_crop2['export_proportion']
### Aggregate Nitrogen embedded in trade by source and destination county
ag_trade_crop2_out = ag_trade_crop2[ag_trade_crop2['Source']!=ag_trade_crop2['Target']].copy()
ag_trade_crop2_import = ag_trade_crop2_out.groupby(['Target', 'commodity']).agg({
     'trade_crop_processing_nitrogen':np.nansum}).reset_index().rename(index=str, 
    columns={'trade_crop_processing_nitrogen':'import_crop_processing_nitrogen', 'Target':'FIPS'})
ag_trade_crop2_export = ag_trade_crop2_out.groupby(['Source', 'commodity']).agg({
     'trade_crop_processing_nitrogen':np.nansum}).reset_index().rename(index=str, 
    columns={'trade_crop_processing_nitrogen':'export_crop_processing_nitrogen', 'Source':'FIPS'})
ag_trade_crop2_self = ag_trade_crop2[ag_trade_crop2['Source']==ag_trade_crop2['Target']].copy()
ag_trade_crop2_self = ag_trade_crop2_self[['FIPS', 'commodity', 'trade_crop_processing_nitrogen']].copy()   
ag_trade_crop2_self = ag_trade_crop2_self.rename(index=str, columns={
    'trade_crop_processing_nitrogen':'selfloop_crop_processing_nitrogen', 'Target':'FIPS'})
### Add Nitrogen embedded in imports and exports to main dataframe
cols1 = ['FIPS', 'commodity', 'export_crop_processing_nitrogen']
cols2 = ['FIPS', 'commodity', 'import_crop_processing_nitrogen']
cols3 = ['FIPS', 'commodity', 'selfloop_crop_processing_nitrogen']
dat = dat.merge(ag_trade_crop2_export[cols1], how='left', on=['FIPS', 'commodity']).merge(
    ag_trade_crop2_import[cols2], how='left', on=['FIPS', 'commodity']).merge(
        ag_trade_crop2_self[cols3], how='left', on=['FIPS', 'commodity'])
dat['import_crop_processing_nitrogen'] = dat['import_crop_processing_nitrogen'].replace(np.nan, 0)
### Account for counties with production that are not present in trade data. These counties
### are treated as self loops with no imports or exports
def func1(row):
    domestic = row['domestic_crop_processing_nitrogen']
    export = row['export_crop_processing_nitrogen']
    selfloop = row['selfloop_crop_processing_nitrogen']
    if (domestic>0) & (np.isnan(export)):
        value = domestic
    else:
        value = selfloop
    return value

dat['selfloop_crop_processing_nitrogen'] = dat.apply(func1, axis=1)
dat['selfloop_crop_processing_nitrogen'] = dat['selfloop_crop_processing_nitrogen'].replace(np.nan, 0)

## Calculate Total Procesed Crop N available for consumption for each crop type (kg N)
dat['processed_crop_nitrogen'] = dat['selfloop_crop_processing_nitrogen'] \
    + dat['import_crop_processing_nitrogen']

## Calculate Weight N content of animal diet for each animal type (%)
dict_diet_composition_broiler = {'corn_grain':40, 'wheat_grain':40, 'soybean':20, 
                            'alfalfa_hay':0, 'corn_silage':0, 'other_hay':0}
dict_diet_composition_layer = {'corn_grain':60, 'wheat_grain':0, 'soybean':40, 
                            'alfalfa_hay':0, 'corn_silage':0, 'other_hay':0}
dict_diet_composition_pig = {'corn_grain':40, 'wheat_grain':40, 'soybean':20, 
                            'alfalfa_hay':0, 'corn_silage':0, 'other_hay':0}
dict_diet_composition_cowbeef = {'corn_grain':40, 'wheat_grain':0, 'soybean':10, 
                            'alfalfa_hay':0, 'corn_silage':10, 'other_hay':40}
dict_diet_composition_cowmilk = {'corn_grain':10, 'wheat_grain':0, 'soybean':10, 
                            'alfalfa_hay':40, 'corn_silage':40, 'other_hay':0}
dat['diet_composition_broiler_ratio'] = dat['commodity'].map(dict_diet_composition_broiler)
dat['diet_composition_layer_ratio'] = dat['commodity'].map(dict_diet_composition_layer)
dat['diet_composition_pig_ratio'] = dat['commodity'].map(dict_diet_composition_pig)
dat['diet_composition_cowbeef_ratio'] = dat['commodity'].map(dict_diet_composition_cowbeef)
dat['diet_composition_cowmilk_ratio'] = dat['commodity'].map(dict_diet_composition_cowmilk)

dat['nitrogen_content_diet_broiler'] = dat['diet_composition_broiler_ratio'] * dat['yield_nitrogen_content']
dat['nitrogen_content_diet_layer'] = dat['diet_composition_layer_ratio'] * dat['yield_nitrogen_content']
dat['nitrogen_content_diet_pig'] = dat['diet_composition_pig_ratio'] * dat['yield_nitrogen_content']
dat['nitrogen_content_diet_cowbeef'] = dat['diet_composition_cowbeef_ratio'] * dat['yield_nitrogen_content']
dat['nitrogen_content_diet_cowmilk'] = dat['diet_composition_cowmilk_ratio'] * dat['yield_nitrogen_content']

cols = ['FIPS', 'county', 'nitrogen_content_diet_broiler', 'nitrogen_content_diet_layer',
        'nitrogen_content_diet_pig', 'nitrogen_content_diet_cowbeef', 'nitrogen_content_diet_cowmilk']
dat2 = dat[cols].groupby(['FIPS', 'county']).sum().reset_index()

dat2['nitrogen_content_diet_broiler'] = dat2['nitrogen_content_diet_broiler']/100
dat2['nitrogen_content_diet_layer'] = dat2['nitrogen_content_diet_layer']/100
dat2['nitrogen_content_diet_pig'] = dat2['nitrogen_content_diet_pig']/100
dat2['nitrogen_content_diet_cowbeef'] = dat2['nitrogen_content_diet_cowbeef']/100
dat2['nitrogen_content_diet_cowmilk'] = dat2['nitrogen_content_diet_cowmilk']/100

## Transform dataframe (for live animals) from wide to long format
dat2 = pd.melt(dat2, id_vars=['FIPS', 'county'], value_vars=['nitrogen_content_diet_broiler', 
                'nitrogen_content_diet_layer', 'nitrogen_content_diet_pig', 
                'nitrogen_content_diet_cowbeef', 'nitrogen_content_diet_cowmilk'])
dat2 = dat2.rename(index=str, columns={'variable':'commodity', 'value':'nitrogen_content_diet'})
dict_columns = {'nitrogen_content_diet_broiler':'broiler', 'nitrogen_content_diet_layer':'layer',
        'nitrogen_content_diet_pig':'pig', 'nitrogen_content_diet_cowbeef':'cow_beef', 
        'nitrogen_content_diet_cowmilk':'cow_milk'}
dat2['commodity'] = dat2['commodity'].map(dict_columns)

## Calculate live animal N uptake for each animal type (%)
animal_nitrogen_content = {'broiler':2.3, 'layer':1.78, 'pig':2, 
                            'cow_beef':2, 'cow_milk':0.5}
feed_conversion_ratio = {'broiler':1.93, 'layer':1.82, 'pig':3.2, 
                            'cow_beef':6.6, 'cow_milk':1}
dat2['animal_nitrogen_content'] = dat2['commodity'].map(animal_nitrogen_content)
dat2['feed_conversion_ratio'] = dat2['commodity'].map(feed_conversion_ratio)

dat2['animal_nitrogen_uptake'] = dat2['animal_nitrogen_content'] / (dat2['feed_conversion_ratio']*dat2['nitrogen_content_diet'])

cols = ['FIPS', 'commodity', 'inventory_head']
dat2 = dat2.merge(inventory_df[cols], how='outer', on=['FIPS', 'commodity'])
dat2 = dat2.dropna(subset=['inventory_head']).copy()

##############################################################################
## Import feed-crop ratio and animalN/feedN ratios data
'''analysis at: codes/crop-animal_transition_methodology3.ipynb '''
feed_crop_ratio = pd.read_csv('../../output/results_2017/crop-animal_transition/feed-crop_ratio_AUcalc.csv', index_col=0)
animalN_feedN_ratio = pd.read_csv('../../output/results_2017/crop-animal_transition/animalN-feedN_ratio_AUcalc.csv', index_col=0)
proportion_animal_units = pd.read_csv('../../output/results_2017/crop-animal_transition/proportion_animal_units_AUcalc.csv', index_col=0)
## Calculate feed nitrogen by crop
dat = dat.merge(feed_crop_ratio, how='left', on=['FIPS', 'commodity'])
dat['feed_nitrogen'] = dat['processed_crop_nitrogen'] * dat['feed-crop_ratio']
## Calculate domestic animal nitrogen
aux = dat.groupby('FIPS').agg({'feed_nitrogen':'sum'}).reset_index()
aux = aux.rename(index=str, columns={'feed_nitrogen': 'feed_nitrogen_total'})
dat2 = dat2.merge(aux, how='left', on='FIPS')
dat2 = dat2.merge(proportion_animal_units, how='left', on=['FIPS', 'commodity'])
dat2 = dat2.merge(animalN_feedN_ratio, how='left', on=['FIPS', 'commodity'])
dat2['feed_nitrogen_calc'] = dat2['proportion_AU'] * dat2['feed_nitrogen_total']
dat2['domestic_animal_nitrogen'] = dat2['animalN-feedN_ratio'] * dat2['feed_nitrogen_calc']
##############################################################################
## Calculate Nitrogen embedded in imports and exports of animals
### Calculate proportion of imports and exports for each link
ag_trade_animal = ag_trade[ag_trade['commodity'].isin(['broiler', 'cow_beef', 
                                        'cow_milk', 'layer', 'pig'])].copy()
ag_trade_animal_export = ag_trade_animal.groupby(['Source', 'commodity']).agg({
    'flow_kg_commodity':np.nansum}).reset_index().rename(index=str, columns={'flow_kg_commodity':'export_kg'})
ag_trade_animal2 = ag_trade_animal.merge(ag_trade_animal_export, how='left', on=['Source', 'commodity'])
ag_trade_animal2['export_proportion'] = ag_trade_animal2['flow_kg_commodity'] / ag_trade_animal2['export_kg']
ag_trade_animal2.dropna(subset=['export_proportion'], inplace=True)
### Add domestic animal processing Nitrogen data to trade data
cols = ['FIPS', 'commodity', 'domestic_animal_nitrogen']
ag_trade_animal2 = ag_trade_animal2.merge(dat2[cols], how='left', left_on=['Source', 'commodity'],
                                          right_on=['FIPS', 'commodity'])
ag_trade_animal2['trade_animal_nitrogen'] = ag_trade_animal2['domestic_animal_nitrogen'] * ag_trade_animal2['export_proportion']
### Aggregate Nitrogen embedded in trade by source and destination county
ag_trade_animal2_out = ag_trade_animal2[ag_trade_animal2['Source']!=ag_trade_animal2['Target']].copy()
ag_trade_animal2_import = ag_trade_animal2_out.groupby(['Target', 'commodity']).agg({
     'trade_animal_nitrogen':np.nansum}).reset_index().rename(index=str, 
    columns={'trade_animal_nitrogen':'import_animal_nitrogen', 'Target':'FIPS'})
ag_trade_animal2_export = ag_trade_animal2_out.groupby(['Source', 'commodity']).agg({
     'trade_animal_nitrogen':np.nansum}).reset_index().rename(index=str, 
    columns={'trade_animal_nitrogen':'export_animal_nitrogen', 'Source':'FIPS'})
ag_trade_animal2_self = ag_trade_animal2[ag_trade_animal2['Source']==ag_trade_animal2['Target']].copy()   
ag_trade_animal2_self = ag_trade_animal2_self[['FIPS', 'commodity', 'trade_animal_nitrogen']].copy()   
ag_trade_animal2_self = ag_trade_animal2_self.rename(index=str, columns={
    'trade_animal_nitrogen':'selfloop_animal_nitrogen', 'Target':'FIPS'})
### Add Nitrogen embedded in imports and exports to main dataframe
cols1 = ['FIPS', 'commodity', 'export_animal_nitrogen']
cols2 = ['FIPS', 'commodity', 'import_animal_nitrogen']
cols3 = ['FIPS', 'commodity', 'selfloop_animal_nitrogen']
dat2 = dat2.merge(ag_trade_animal2_export[cols1], how='left', on=['FIPS', 'commodity']).merge(
    ag_trade_animal2_import[cols2], how='left', on=['FIPS', 'commodity']).merge(
        ag_trade_animal2_self[cols3], how='left', on=['FIPS', 'commodity'])
dat2['import_animal_nitrogen'] = dat2['import_animal_nitrogen'].replace(np.nan, 0)
### Account for counties with production that are not present in trade data. These counties
### are treated as self loops with no imports or exports
def func1(row):
    domestic = row['domestic_animal_nitrogen']
    export = row['export_animal_nitrogen']
    selfloop = row['selfloop_animal_nitrogen']
    if (domestic>0) & (np.isnan(export)):
        value = domestic
    else:
        value = selfloop
    return value

dat2['selfloop_animal_nitrogen'] = dat2.apply(func1, axis=1)
dat2['selfloop_animal_nitrogen'] = dat2['selfloop_animal_nitrogen'].replace(np.nan, 0)

## Calculate Total Live Animal N weight (to be slaughtered/milked/laid) for each animal type (Kg N)
dat2['animal_nitrogen'] = dat2['selfloop_animal_nitrogen'] + dat2['import_animal_nitrogen'] 

## Calculate crop diet composition N for each animal type (%)
yn = [i for i in dict_yield_N_content.values()]
list1 = [dict_diet_composition_broiler, dict_diet_composition_layer,
       dict_diet_composition_pig, dict_diet_composition_cowbeef,
       dict_diet_composition_cowmilk]
list_final = []
for d in list1:
    dc = [i for i in d.values()]
    mult = np.multiply(yn, dc)
    mult_sum = np.sum(mult)
    ratio = mult/mult_sum
    list_final.append(ratio)

anim_list = ['broiler', 'layer', 'pig', 'cow_beef', 'cow_milk']
dict_crop_diet_percentage_corn = dict(zip(anim_list, [i[0] for i in list_final]))
dict_crop_diet_percentage_wheat = dict(zip(anim_list, [i[1] for i in list_final]))
dict_crop_diet_percentage_soybean = dict(zip(anim_list, [i[2] for i in list_final]))
dict_crop_diet_percentage_alfalfa = dict(zip(anim_list, [i[3] for i in list_final]))
dict_crop_diet_percentage_cornsilage = dict(zip(anim_list, [i[4] for i in list_final]))
dict_crop_diet_percentage_otherhay = dict(zip(anim_list, [i[5] for i in list_final]))

dat2['crop_diet_percentage_corn'] = dat2['commodity'].map(dict_crop_diet_percentage_corn)
dat2['crop_diet_percentage_wheat'] = dat2['commodity'].map(dict_crop_diet_percentage_wheat)
dat2['crop_diet_percentage_soybean'] = dat2['commodity'].map(dict_crop_diet_percentage_soybean)
dat2['crop_diet_percentage_alfalfa'] = dat2['commodity'].map(dict_crop_diet_percentage_alfalfa)
dat2['crop_diet_percentage_cornsilage'] = dat2['commodity'].map(dict_crop_diet_percentage_cornsilage)
dat2['crop_diet_percentage_otherhay'] = dat2['commodity'].map(dict_crop_diet_percentage_otherhay)

## Calculate Crop diet composition N weight for each animal type (Kg N)
dat2['nitrogen_content_diet_weight_corn'] = dat2['crop_diet_percentage_corn'] * dat2['feed_nitrogen_calc']
dat2['nitrogen_content_diet_weight_wheat'] = dat2['crop_diet_percentage_wheat'] * dat2['feed_nitrogen_calc']
dat2['nitrogen_content_diet_weight_soybean'] = dat2['crop_diet_percentage_soybean'] * dat2['feed_nitrogen_calc']
dat2['nitrogen_content_diet_weight_alfalfa'] = dat2['crop_diet_percentage_alfalfa'] * dat2['feed_nitrogen_calc']
dat2['nitrogen_content_diet_weight_cornsilage'] = dat2['crop_diet_percentage_cornsilage'] * dat2['feed_nitrogen_calc']
dat2['nitrogen_content_diet_weight_otherhay'] = dat2['crop_diet_percentage_otherhay'] * dat2['feed_nitrogen_calc']

## Calculate Animal Carcass/milked/laid N for each animal type (Kg N)
dict_animal_slaughtering_rate = {'broiler':0.75, 'layer':0.95, 'pig':0.75, 
                                 'cow_beef':0.75, 'cow_milk':0.98}
dat2['animal_slaughtering_rate'] = dat2['commodity'].map(dict_animal_slaughtering_rate)
dat2['animal_carcass_milked_laid_nitrogen'] = dat2['animal_slaughtering_rate'] * dat2['animal_nitrogen']

## Calculate Recycled N of Slaughtering/milking/laying waste from each animal byproducts to serve as animal feed for that animal type (Kg N)
dict_nitrogen_recycling_rate3 = {'broiler':0.9, 'layer':0.9, 'pig':0.9, 
                                 'cow_beef':0.9, 'cow_milk':0.9}
dat2['nitrogen_recycling_rate3'] = dat2['commodity'].map(dict_nitrogen_recycling_rate3)
dat2['nitrogen_recycling3'] = (dat2['animal_nitrogen'] - dat2['animal_carcass_milked_laid_nitrogen']) \
                                * dat2['nitrogen_recycling_rate3']

# Calculate Slaughtering/milking/laying N loss to the environment for each animal type (Kg N)
dat2['nitrogen_loss4'] = (dat2['animal_nitrogen'] - dat2['animal_carcass_milked_laid_nitrogen']) \
                                - dat2['nitrogen_recycling3']

## Calculate Domestic Animal Product N for each animal type (Kg N)
dict_animal_product_processing_rate = {'broiler':0.9, 'layer':0.84, 'pig':0.9, 
                                       'cow_beef':0.9, 'cow_milk':0.98}
dat2['animal_product_processing_rate'] = dat2['commodity'].map(dict_animal_product_processing_rate)
dat2['domestic_animal_product_nitrogen'] = dat2['animal_carcass_milked_laid_nitrogen'] * dat2['animal_product_processing_rate']

## Calculate Recycled N of Animal Product Processing waste from each animal byproducts to serve as animal feed for that animal type (Kg N)
dict_nitrogen_recycling_rate4 = {'broiler':0.9, 'layer':0.9, 'pig':0.9, 
                                 'cow_beef':0.9, 'cow_milk':0.9}
dat2['nitrogen_recycling_rate4'] = dat2['commodity'].map(dict_nitrogen_recycling_rate4)
dat2['nitrogen_recycling4'] = (dat2['animal_carcass_milked_laid_nitrogen'] - dat2['domestic_animal_product_nitrogen']) \
                                * dat2['nitrogen_recycling_rate4']
                                
## Calculate Animal Product Processing N loss to the environment from each animal type (Kg N)
dat2['nitrogen_loss5'] = (dat2['animal_carcass_milked_laid_nitrogen'] - dat2['domestic_animal_product_nitrogen']) \
                                - dat2['nitrogen_recycling4']

## Calculate Nitrogen embedded in imports and exports of meats
### Calculate proportion of imports and exports for each link
ag_trade_meat = meat_trade[meat_trade['commodity'].isin(['broiler', 'cow_beef', 
                                        'cow_milk', 'layer', 'pig'])].copy()
ag_trade_meat_export = ag_trade_meat.groupby(['Source', 'commodity']).agg({
    'flow_kg_commodity':np.nansum}).reset_index().rename(index=str, columns={'flow_kg_commodity':'export_kg'})
ag_trade_meat2 = ag_trade_meat.merge(ag_trade_meat_export, how='left', on=['Source', 'commodity'])
ag_trade_meat2['export_proportion'] = ag_trade_meat2['flow_kg_commodity'] / ag_trade_meat2['export_kg']
ag_trade_meat2.dropna(subset=['export_proportion'], inplace=True)
### Add domestic meat processing Nitrogen data to trade data
cols = ['FIPS', 'commodity', 'domestic_animal_product_nitrogen']
ag_trade_meat2 = ag_trade_meat2.merge(dat2[cols], how='left', 
                    left_on=['Source', 'commodity'], right_on=['FIPS', 'commodity'])
ag_trade_meat2['trade_meat_nitrogen'] = ag_trade_meat2['domestic_animal_product_nitrogen'] * ag_trade_meat2['export_proportion']
### Aggregate Nitrogen embedded in trade by source and destination county
ag_trade_meat2_out = ag_trade_meat2[ag_trade_meat2['Source']!=ag_trade_meat2['Target']].copy()
ag_trade_meat2_import = ag_trade_meat2_out.groupby(['Target', 'commodity']).agg({
     'trade_meat_nitrogen':np.nansum}).reset_index().rename(index=str, 
    columns={'trade_meat_nitrogen':'import_meat_nitrogen', 'Target':'FIPS'})
ag_trade_meat2_export = ag_trade_meat2_out.groupby(['Source', 'commodity']).agg({
     'trade_meat_nitrogen':np.nansum}).reset_index().rename(index=str, 
    columns={'trade_meat_nitrogen':'export_meat_nitrogen', 'Source':'FIPS'})
ag_trade_meat2_self = ag_trade_meat2[ag_trade_meat2['Source']==ag_trade_meat2['Target']].copy()   
ag_trade_meat2_self = ag_trade_meat2_self[['FIPS', 'commodity', 'trade_meat_nitrogen']].copy()   
ag_trade_meat2_self = ag_trade_meat2_self.rename(index=str, columns={
    'trade_meat_nitrogen':'selfloop_meat_nitrogen', 'Target':'FIPS'})
### Add Nitrogen embedded in imports and exports to main dataframe
cols1 = ['FIPS', 'commodity', 'export_meat_nitrogen']
cols2 = ['FIPS', 'commodity', 'import_meat_nitrogen']
cols3 = ['FIPS', 'commodity', 'selfloop_meat_nitrogen']
dat2 = dat2.merge(ag_trade_meat2_export[cols1], how='left', on=['FIPS', 'commodity']).merge(
    ag_trade_meat2_import[cols2], how='left', on=['FIPS', 'commodity']).merge(
        ag_trade_meat2_self[cols3], how='left', on=['FIPS', 'commodity'])
dat2['import_meat_nitrogen'] = dat2['import_meat_nitrogen'].replace(np.nan, 0)
### Account for counties with production that are not present in trade data. These counties
### are treated as self loops with no imports or exports
def func1(row):
    domestic = row['domestic_animal_product_nitrogen']
    export = row['export_meat_nitrogen']
    selfloop = row['selfloop_meat_nitrogen']
    if (domestic>0) & (np.isnan(export)):
        value = domestic
    else:
        value = selfloop
    return value

dat2['selfloop_meat_nitrogen'] = dat2.apply(func1, axis=1)
dat2['selfloop_meat_nitrogen'] = dat2['selfloop_meat_nitrogen'].replace(np.nan, 0)

## Calculate Total Animal Product N (to be consumed by humans) for each animal type (Kg N)
dat2['animal_product_nitrogen'] = dat2['selfloop_meat_nitrogen'] + dat2['import_meat_nitrogen']

## Calculate Animal Product N Consumption for each animal type (Kg N)
dict_animal_product_consumption_rate = {'broiler':0.84, 'layer':0.98, 'pig':0.84, 
                                        'cow_beef':0.84, 'cow_milk':0.68}
dat2['animal_product_consumption_rate'] = dat2['commodity'].map(dict_animal_product_consumption_rate)
dat2['animal_product_nitrogen_consumption'] = dat2['animal_product_nitrogen'] * dat2['animal_product_consumption_rate']

## Calculate Animal Product N consumption loss to the environment  from each animal type (Kg N)
dat2['nitrogen_loss6'] = dat2['animal_product_nitrogen'] - dat2['animal_product_nitrogen_consumption']

## Calculate N loss to the environment from Human waste of Animal Product Consumption from each animal type (Kg N)
dat2['nitrogen_loss7'] = dat2['animal_product_nitrogen_consumption']

## Calculate Recycled N as animal feed to each animal type from Slaughtering-milking-laying and Food processing (Kg N)
### from Slaughtering-milking-laying (Kg N)
dat2['feed_nitrogen_recycled_slaughtering_waste'] = dat2['nitrogen_recycling3']
### from Food processing (Kg N)
dat2['feed_nitrogen_recycled_meat_waste'] = dat2['nitrogen_recycling4']

## Calculate Total feed N to each animal type from both new and recycled sources (Kg N)
dat2['total_feed_nitrogen'] = dat2['feed_nitrogen_calc'] + dat2['feed_nitrogen_recycled_slaughtering_waste'] \
                            + dat2['feed_nitrogen_recycled_meat_waste']

## Calculate Manure loss and feed waste  from each animal type (Kg N)
dat2['manure_feed_nitrogen_waste'] = dat2['total_feed_nitrogen'] - dat2['domestic_animal_nitrogen']

## Calculate recycled Manure loss and feed waste from each animal type (Kg N)
dict_nitrogen_recycling_rate2 = {'broiler':0.35, 'layer':0.35, 'pig':0.35, 
                                 'cow_beef':0.35, 'cow_milk':0.35}
dat2['nitrogen_recycling_rate2'] = dat2['commodity'].map(dict_nitrogen_recycling_rate2)
dat2['nitrogen_recycling2'] = dat2['manure_feed_nitrogen_waste'] * dat2['nitrogen_recycling_rate2']
R2_total = dat2[['FIPS', 'nitrogen_recycling2']].groupby('FIPS').sum().reset_index().rename(
    index=str, columns={'nitrogen_recycling2':'nitrogen_recycling2_total'})

#Calculate Manure loss and feed waste to the environemnt from each animal type (Kg N)
dat2['nitrogen_loss3'] = dat2['manure_feed_nitrogen_waste'] - dat2['nitrogen_recycling2']
dat2['nitrogen_loss3'] = dat2['nitrogen_loss3'].apply(lambda x:0 if x<0 else x)

## Calculate Recycled Input N to each Crop type from Crop Residue and Manure(Kg N)
### ratio of Recycled N (RNr) Share for each crop type
aux = dat[['FIPS', 'commodity', 'fertilizer_nitrogen', 'domestic_crop_nitrogen']].copy()
aux['subtraction'] = aux['domestic_crop_nitrogen'] - aux['fertilizer_nitrogen']
aux2 = aux[aux['commodity'].isin(['corn_grain', 'corn_silage', 'wheat_grain'])]
aux2_sum = aux2[['FIPS', 'subtraction']].groupby('FIPS').sum().reset_index().rename(
    index=str, columns={'subtraction':'subtraction_sum'})
aux = aux.merge(aux2_sum[['FIPS', 'subtraction_sum']], how='left', on='FIPS')
aux['recycled_nitrogen_crop_share'] = aux['subtraction'] / aux['subtraction_sum']
#### recycled nitrogen not for soybean, hay and other hay
aux['boolean'] = aux['commodity'].apply(lambda x:1 if x in ['corn_grain', 'wheat_grain', 'corn_silage'] else 0)
aux['recycled_nitrogen_crop_share'] = aux['recycled_nitrogen_crop_share'] * aux['boolean'] 
dat = dat.merge(aux[['FIPS', 'commodity', 'recycled_nitrogen_crop_share']], how='left', on=['FIPS', 'commodity'])
### From Crop Residue (Kg N)
dat = dat.merge(R1_total, how='left', on='FIPS')
dat['nitrogen_recycling1_from_crop'] = dat['recycled_nitrogen_crop_share'] * dat['nitrogen_recycling1_total']
### From Manure (Kg N)
dat = dat.merge(R2_total, how='left', on='FIPS')
dat['nitrogen_recycling2_from_manure'] = dat['recycled_nitrogen_crop_share'] * dat['nitrogen_recycling2_total']
##############################################################################
#### Check if recycled manure is higher than limit (limits are calculated in check_maure_application_rate.py)
dat['recycled_manure_appl_rate'] = dat['nitrogen_recycling2_from_manure'] *2.2 / dat['harvested_area_acre']
dict_limit_manure_rate = {'corn_grain':257.6, 'wheat_grain':663.6, 'soybean':0, 
                          'alfalfa_hay':0, 'corn_silage':312.8, 'other_hay':0}
dat['limit_recycled_manure_appl_rate'] = dat['commodity'].map(dict_limit_manure_rate)
dat['nitrogen_recycling2_from_manure_limit'] = dat.apply(lambda x:x.nitrogen_recycling2_from_manure 
                        if x.recycled_manure_appl_rate < x.limit_recycled_manure_appl_rate 
                        else x.limit_recycled_manure_appl_rate*x.harvested_area_acre/2.2, axis=1)
dat['excess_recycled_nitrogen_manure'] = dat['nitrogen_recycling2_from_manure'] - dat['nitrogen_recycling2_from_manure_limit']
R2_excess_total = dat.groupby('FIPS').agg({'excess_recycled_nitrogen_manure':'sum'}).reset_index().rename(
    index=str, columns={'excess_recycled_nitrogen_manure':'excess_recycled_nitrogen_manure_total'})
dat2['proportion_manure_feed_nitrogen_waste'] = dat2['manure_feed_nitrogen_waste'] / dat2.groupby('FIPS')['manure_feed_nitrogen_waste'].transform('sum')
dat2 = dat2.merge(R2_excess_total, how='left', on='FIPS')
dat2['excess_recycled_nitrogen_manure'] = dat2['proportion_manure_feed_nitrogen_waste'] * dat2['excess_recycled_nitrogen_manure_total']
dat2['nitrogen_loss3'] = dat2['nitrogen_loss3'] + dat2['excess_recycled_nitrogen_manure']

## Calculate Total Recycled input N added to each crop type from both Crop residue and Manure (kg N)
dat['total_recycled_nitrogen_input'] = dat['nitrogen_recycling1_from_crop'] + dat['nitrogen_recycling2_from_manure_limit'] 
##############################################################################

## Calculate Total input N added to each crop type from both New and Recycled sources (kg N)
dat['total_nitrogen_input'] = dat['total_new_input_nitrogen'] + dat['total_recycled_nitrogen_input']

## Calculate N loss in Stage 1: N Input not taken up by crops to the environment for each crop type (kg N)
dat['nitrogen_loss1'] = dat['total_nitrogen_input'] - dat['domestic_crop_nitrogen']
dat['nitrogen_loss1'] = dat['nitrogen_loss1'].apply(lambda x:0 if x<0 else x)
# Calculate total nitrogen input
dat['total_nitrogen_input'] = dat['fertilizer_nitrogen'] + dat['biological_nitrogen_fixation']

# Calculate nitrogen in harvested crops
dat['nitrogen_in_harvest'] = dat['yield_nitrogen'] * dat['harvested_area']

# Calculate NUE
dat['nue'] = dat['nitrogen_in_harvest'] / dat['total_nitrogen_input']

# Calculate overall NUE for the entire dataset
overall_nue = dat['nitrogen_in_harvest'].sum() / dat['total_nitrogen_input'].sum()

print(f"Overall Nitrogen Use Efficiency: {overall_nue:.2f}")


dat2['feed_to_inventory_head_ratio'] = dat2['inventory_head'] / dat2['feed_nitrogen_calc']

# Save to csv
## Select columns from dataframes
cols0 = ['FIPS', 'county', 'commodity']
cols = ['domestic_crop_nitrogen', 'processed_crop_nitrogen', 'nitrogen_loss1', 'nitrogen_loss2']
dat_select = dat[cols0 + cols].copy()
cols = ['domestic_animal_nitrogen', 'animal_nitrogen', 'feed_nitrogen_calc',
        'animal_carcass_milked_laid_nitrogen', 'animal_product_nitrogen',
        'animal_product_nitrogen_consumption', 'nitrogen_loss3', 'nitrogen_loss4',
        'nitrogen_loss5', 'nitrogen_loss6', 'nitrogen_loss7']
dat2_select = dat2[cols0 + cols].copy()
desired_dir_path = r'C:\Users\%s\Box\AT Research\AG\Model\codeshare\codeshare\output\results_2017\nitrogen_flow_model_v4'%user
full_path = os.path.join(desired_dir_path, folder_name_output)
if not os.path.exists(full_path):
    os.makedirs(full_path)
dat_select.to_csv(os.path.join(full_path, 'crop_interlayer_flows.csv'))
dat.to_csv(os.path.join(full_path, 'crop_processing_nitrogen.csv'))
dat2.to_csv(os.path.join(full_path, 'animal_stage_nitrogen.csv'))
# dat_select.to_csv('../../output/results_2017/nitrogen_flow_model_v4/%s/crop_interlayer_flows.csv'%folder_name_output)
dat2_select.to_csv('../../output/results_2017/nitrogen_flow_model_v4/%s/meat_interlayer_flows.csv' % folder_name_output)

cols = ['Source', 'Target', 'commodity', 'trade_animal_nitrogen']
ag_trade_animal2[cols].to_csv(
    '../../output/results_2017/nitrogen_flow_model_v4/%s/animal_intralayer_flows.csv' % folder_name_output)
cols = ['Source', 'Target', 'commodity', 'trade_crop_processing_nitrogen']
ag_trade_crop2[cols].to_csv(
    '../../output/results_2017/nitrogen_flow_model_v4/%s/crop_intralayer_flows.csv' % folder_name_output)
cols = ['Source', 'Target', 'commodity', 'trade_meat_nitrogen']
ag_trade_meat2[cols].to_csv(
    '../../output/results_2017/nitrogen_flow_model_v4/%s/meat_intralayer_flows.csv' % folder_name_output)

