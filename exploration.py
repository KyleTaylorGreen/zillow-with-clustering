import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler
import acquire
import prepare
import split
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats
import gmaps
import env

from IPython.display import display

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

alpha = 0.05
fig_size = (10,7)

def na_summary(df):
    names = [df.columns]
    
    for name in names:
        mean_of_na = pd.DataFrame(df[name].isna().mean())
        sum_of_na = pd.DataFrame(df[name].isna().sum())
        merged = pd.merge(sum_of_na,mean_of_na, left_index=True, right_index=True)
        merged = merged.rename(columns={'0_x': 'num_rows_missing', '0_y': 'pct_rows_missing'})
        
    return merged

def create_subplots(quant_cols, single_var=True):
    subplot_dim = find_subplot_dim(quant_cols)
    plots= []
    fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], figsize=(fig_size[0], fig_size[1]))
    axes = axes.flatten()
    
    for axe in axes:
        plots.append(axe)
    
    print(f'Plots: {plots}')
    return plots, fig


def quantitative_hist_boxplot_describe(training_df, quantitative_col_names,separate=True):
    plots, fig = create_subplots(quantitative_col_names)
    for i, col in enumerate(quantitative_col_names):
        plots[i].hist(training_df[col])
        plots[i].set_xlabel(col)
        plots[i].set_ylabel('count')
    plt.show()
    
    if separate:
        print(type(training_df))
        plots, axes = create_subplots(quantitative_col_names)
        for i, col in enumerate(quantitative_col_names):
            training_df.boxplot(ax=plots[i], column=col)
        plt.show()
    else:    
        training_df.boxplot(column=quantitative_col_names)
        plt.show()

    print(training_df[quantitative_col_names].describe().T)
    

def target_freq_hist_count(training_df, target_col):
    freq_hist = training_df[target_col].hist()
    #print(training_df[target_col].value_counts())
    #plt.show()
    return freq_hist

def odd(num):
    if num % 2 != 0:
        return True
    else:
        return False
    
def even(num):
    return not odd(num)

def find_subplot_dim(quant_col_lst):
    
    # goal: make x 
    # checks if len is even (making 2 rows)
    if even(len(quant_col_lst)):
        length = len(quant_col_lst)
    else:
        length = len(quant_col_lst) + 1
        
    divided_by_2 = int(length/ 2)
    divided_by_other_factor = int(length / divided_by_2)
    subplot_dim = [divided_by_2, divided_by_other_factor]
    
    return subplot_dim

def quant_vs_target_bar(train_df, target_col, quant_col_lst, mean_line=False):
    
    subplot_dim = find_subplot_dim(quant_col_lst)
    
    plots = []
    fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], sharex=True, figsize=(10,5))
    
    axes = axes.flatten()
    
    for axe in axes:
        plots.append(axe)

    for n in range(len(quant_col_lst)):    
        sns.barplot(ax=plots[n], x=train_df[target_col], y =train_df[quant_col_lst[n]])
        
        if mean_line:
            avg = train_df[quant_col_lst[n]].mean()
            plots[n].axhline(avg,  label=f'Avg {train_df[quant_col_lst[n]]}')

def describe_quant_grouped_by_target(training_df, quantitative_col, 
                                     target_col):
    lst_cpy = quantitative_col[:]
    lst_cpy.append(target_col)
    
    print(training_df[lst_cpy].groupby(target_col).describe().T)


def target_subsets(target_col, training_df):
    
    values = training_df[target_col].unique()
    subset_dict= {}
    
    for val in values:
        subset_dict[val] = training_df[training_df[target_col]==val]
        
    return subset_dict

def combinations_of_subsets(target_col, training_df):
    subsets = target_subsets(target_col, training_df)
    combos = list(itertools.combinations(subsets.keys(), 2))
    
    return subsets, combos

def mannshitneyu_for_quant_by_target(target_col, training_df, 
                                    quantitative_col):
    
    predictors = {}
    subsets, combos = combinations_of_subsets(target_col, training_df)
    p_exceeds_alpha = []
        

    for i, pair in enumerate(combos):
        
        #print(f'{pair[0]}/{pair[1]}:' )
        predictors[str(pair)] = []
        for col in quantitative_col:
            #print(subsets[pair[0]][col])
            t, p = stats.mannwhitneyu(subsets[pair[0]][col], 
                                      subsets[pair[1]][col])
            #print(f'{pair[0]}/{pair[1]} {col}:')
            #print(f't: {t}, p: {p}\n')
            
            if p < alpha:
                predictors[str(pair)].append({col: [t, p]})
            else:
                p_exceeds_alpha.append([str(pair), col, t, p])
                
                
    return subsets, predictors, p_exceeds_alpha, combos
            
    
def print_mannswhitneyu_predictors(predictors):
    for keys, values in predictors.items():
        print(keys)
        for value in values:
            print(value)
        print()
    
def print_mannswhitneyu_failures(p_exceeds_alpha):
    for val in p_exceeds_alpha:
        print(f'Combination: {val[0]}')
        print(f'Measurement: {val[1]}')
        print(f't: {val[2]}, p: {val[3]}')
        print()
        

def print_quant_by_target(target_col, training_df, quant_col):
    subsets, predictors, p_exceeds_alpha, combos = mannshitneyu_for_quant_by_target(target_col, 
                                                                            training_df, 
                                                                            quant_col)
    print_mannswhitneyu_predictors(predictors)
    print_mannswhitneyu_failures(p_exceeds_alpha)
    
    combo_predic = {}
    for combo in combos:
        combo_predic[combo] = []
        #print(predictors[str(combo)])
        for predic in predictors[str(combo)]:
              #print(list(predic.keys())[0])
              combo_predic[combo].append(list(predic.keys())[0])

    return subsets, predictors, p_exceeds_alpha, combo_predic

def two_quants_by_target_var(target_col, training_df, combo_predic, 
                         subtitle=""):
    
    for combo in combo_predic.keys():
        subplot_dim= find_subplot_dim(combo_predic[combo])
        
        plots = []
        fig, axes = plt.subplots(subplot_dim[0], subplot_dim[1], figsize=(fig_size[0],fig_size[1]))
        
        axes = axes.flatten()
    
        for axe in axes:
            plots.append(axe)
                
        predictors_comb = list(itertools.combinations(combo_predic[combo], 2))
    
        for i, pair in enumerate(predictors_comb):
            sns.scatterplot(x=training_df[pair[0]], y=training_df[pair[1]],
                           hue=training_df[target_col],
                           ax= plots[i])
            plots[i].set_xlabel(pair[0])
            plots[i].set_ylabel(pair[1])
        plt.show()

def categorical_comparisons(train_df, categories, target_var):
    categories_comb = list(itertools.combinations(categories, 2))
    churn_related = [comb for comb in categories_comb if target_var in comb]

    result_dicts = {}
    for comb in churn_related:
        observed = pd.crosstab(train_df[comb[0]], train_df[comb[1]])
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        result_dicts[comb] = [chi2, p, degf, expected]

    return result_dicts

def filter_category_compar_results(train_df, categories, target_var):
    results = categorical_comparisons(train_df, categories, target_var)
    filtered = {}

    for key, values in results.items():
        if values[1] < alpha and values[1] != 0:
            filtered.update({key: values})

    sorted_target = []
    for item in filtered.keys():
        item = list(item)
        item.remove(target_var)
        sorted_target.append(item[0])

    print(f"Categories related to {target_var}:")
    for elem in sorted_target:
        print(elem)
    
    return sorted_target

def dataset_reduction(train_df, target_var, categories, quant_cols):
    cats_related_to_target = filter_category_compar_results(train_df, categories, target_var)
    final_df = ['customer_id']
    for col in quant_cols:
        final_df.append(col)
    for cat in cats_related_to_target:
        final_df.append(cat)
    final_df.append(target_var)

    final_df = train_df[final_df]

    return final_df


def overview(train_df, categories,
             quant_cols, target_var):
    quantitative_hist_boxplot_describe(train_df, quant_cols,separate=True)
    target_freq_hist_count(train_df, target_var)
    quant_vs_target_bar(train_df, target_var, quant_cols, mean_line=True)
    describe_quant_grouped_by_target(train_df, quant_cols, target_var)

    subsets, predictors, p_exceeds_alpha, combos = print_quant_by_target(target_var, train_df, quant_cols)
    two_quants_by_target_var(target_var, train_df, combos)
    
    sns.pairplot(train_df, hue=target_var)
    final_df = dataset_reduction(train_df,
                                target_var,
                                categories,
                                quant_cols)

    return final_df

def plot_variable_pairs(train, quants, sample=10_000):
    
    for pair in list(itertools.combinations(train[quants].columns, 2)):
        sns.lmplot(x=pair[0], y=pair[1], data=train.sample(sample), hue='county')

def zillow_univariate_hists(df):
    cols_to_visualize = df.drop(columns='parcelid').columns
    for col in cols_to_visualize:
        target_freq_hist_count(df, col)
        plt.xlabel(col)
        plt.show()

def plot_categorical_and_continuous_vars(df, continuous, cats):
    combos = []
    for cat in cats:
        for cont in continuous:
            combos.append([cont, cat])
        
    for combo in combos:
        sns.violinplot(x=combo[0], y=combo[1], data=df.sample(1_000))
        plt.show()

def add_back_to_train(train, df, col_to_add, shared_id_column):
    

    col_values = []

    for value in train[shared_id_column]:
        col_values.append(df.query(f'{shared_id_column} == {value}')[f'{col_to_add}'].values[0])

    train[f'{col_to_add}'] = col_values

    return train

def a_matrix():
    actual = [1,1,0,0, 1,0,0,0,0,0,0]
    pred = [0,0,0,0,0,0,0,0,0,0,0]

    results = confusion_matrix(actual, pred)
    print(results)
    print(classification_report(actual,pred))

def county_borders(df, resolution):
    zillow = df.copy()
    border_dict = {}
    grids = []
    prices = []
    
    for county in zillow.county.unique():
        border_dict[f'{county}_lat_min'] = zillow[zillow.county == county].latitude.min()
        border_dict[f'{county}_lat_max'] = zillow[zillow.county == county].latitude.max()
        border_dict[f'{county}_lon_min'] = zillow[zillow.county == county].longitude.min()
        border_dict[f'{county}_lon_max'] = zillow[zillow.county == county].longitude.max()
        
        #print(border_dict)
        
        border_dict = lat_lon_steps(border_dict, zillow, resolution, county)
        
        border_dict[f'{county}_lat'] = np.arange(border_dict[f'{county}_lat_min'], 
                                                 border_dict[f'{county}_lat_max'],
                                                 border_dict[f'{county}_lat_step'])
        
        
        border_dict[f'{county}_lon'] = np.arange(border_dict[f'{county}_lon_min'],
                                                 border_dict[f'{county}_lon_max'],
                                                 border_dict[f'{county}_lon_step'])
        
    return border_dict
    


def lat_lon_steps(border_dict, zillow, resolution, county):
    
    border_dict[f'{county}_lat_step'] = (border_dict[f'{county}_lat_max'] - border_dict[f'{county}_lat_min']) / resolution
    border_dict[f'{county}_lon_step'] = (border_dict[f'{county}_lon_max'] - border_dict[f'{county}_lon_min']) / resolution

    return border_dict

def create_grids(border_dict, resolution, df, sample=10_000):
    grids = []
    prices = []
    lat_coords = []
    lon_coords = []
    sqr_ft = []
    elevation  = []
    lotsizesqrft = []
    added = 0
    
    if sample:
        zillow = df.sample(sample)
        
    else:
        zillow = df.copy()
    
    for i, county in enumerate(np.sort(zillow.county.unique())):
        grids.append(np.zeros((resolution, resolution)))
        prices.append(np.zeros((resolution, resolution)))
        lat_coords.append(np.zeros((resolution,resolution)))
        lon_coords.append(np.zeros((resolution,resolution)))
        sqr_ft.append(np.zeros((resolution,resolution)))
        elevation.append(np.zeros((resolution, resolution)))
        lotsizesqrft.append(np.zeros((resolution, resolution)))
        
        county_subset = zillow[zillow.county == county]
        print(county)
        
        for a in range(len(county_subset)):
            
            for b1 in range(resolution):
                if border_dict[f'{county}_lat'][b1] - (border_dict[f'{county}_lat_step']/2) <= county_subset.latitude.values[a] < border_dict[f'{county}_lat'][b1] + (border_dict[f'{county}_lat_step'] /2):
                
                    for b2 in range(resolution):
                        if border_dict[f'{county}_lon'][b2] - (border_dict[f'{county}_lon_step']/2) <= county_subset.longitude.values[a] < border_dict[f'{county}_lon'][b2] + (border_dict[f'{county}_lon_step']/2):

                            prices[i][b1, b2] += county_subset.taxvaluedollarcnt.values[a]
                            grids[i][b1, b2] += 1
                            sqr_ft[i][b1,b2] += county_subset.sqr_ft.values[a]
                            lat_coords[i][b1, b2] = border_dict[f'{county}_lat'][b1]
                            lon_coords[i][b1, b2] = border_dict[f'{county}_lon'][b2]
                            elevation[i][b1, b2] += county_subset.elevation.values[a]
                            lotsizesqrft[i][b1,b2] += county_subset.lotsizesquarefeet.values[a]
                            break
                    break
                            
            added += 1
            if added % (sample/5) == 0:
                print(added)
        #print(grids[i])
    
    return grids, prices, lat_coords, lon_coords, sqr_ft, resolution, elevation, lotsizesqrft


def create_map_data(df, resolution, sample=10_000):
    border_dict = county_borders(df, resolution)
    grids, prices, lat_coords, lon_coords, sqr_ft, resolution, elevation, lotsizesqrft = create_grids(border_dict, resolution, 
                                                                                 df, sample=sample)
    
    return prices, grids, lat_coords, lon_coords, sqr_ft, resolution, elevation, lotsizesqrft
  

def create_display_heatmap(prices, grids, lat_coords, lon_coords, sqr_ft, resolution, elevation,
                           lotsizesqrft,
                           counties=['la'], option='average_price'):
    
    county_dfs = []
    
    api_key = env.gmaps_api()
    gmaps.configure(api_key=api_key)
    
    for county in counties:
        i = 0
        if county == 'oc':
            i = 1
        elif county == 'ventura':
            i = 2
        
        county_latitude_values = lat_coords[i].reshape((resolution**2,))
        county_longitude_values = lon_coords[i].reshape((resolution**2,))
        county_prices = prices[i].reshape((resolution**2,))
        county_grid = grids[i].reshape((resolution**2,))
        county_sqr_ft = sqr_ft[i].reshape((resolution**2,))
        county_elevation = elevation[i].reshape((resolution**2,))
        county_lotsizesqrft = lotsizesqrft[i].reshape((resolution**2,))


        heatmap_prices = {'county_prices': county_prices, 'latitude': county_latitude_values, 
                          'longitude': county_longitude_values, 'num_houses': county_grid,
                          'sqr_ft': county_sqr_ft, 'elevation': county_elevation,
                          'lotsizesqrft': county_lotsizesqrft}

        county_df =pd.DataFrame(heatmap_prices)
        county_df = county_df[county_df.num_houses > 2]

        county_df['avg_elevation'] = county_df.elevation / county_df.num_houses
        county_df['avg_prices'] = county_df.county_prices / county_df.num_houses
        county_df['price_per_sqr_ft'] = county_df.county_prices / county_df.sqr_ft
        county_df['avg_lotsizesqrft'] = county_df.lotsizesqrft / county_df.num_houses
        print(county_df.price_per_sqr_ft.max())
        
        county_dfs.append(county_df)

        locations = county_df[['latitude', 'longitude']]
        
        options = {'average_price': county_df['avg_prices'], 'price_per_sqr_ft': county_df['price_per_sqr_ft']}
        weights = options[option]

        fig = gmaps.figure()
        heatmap_layer = gmaps.heatmap_layer(locations=locations, weights=weights, 
                                            max_intensity=weights.max(), point_radius=6.1)

        fig.add_layer(heatmap_layer)
        display(fig)
        
    return county_dfs
    
def pearsonr_quants(quants, df):
    combos = list(itertools.combinations(quants, 2))
    results_dict = {combo: stats.pearsonr(df[combo[0]], df[combo[1]])\
                    for combo in combos}
    for combo in combos:
        print(f'{combo[0]}:        {combo[1]}:')
        print(stats.pearsonr(df[combo[0]], df[combo[1]]))

    return results_dict

def pearsonr_remove_non_target_var(results_dict, target_var):

    for keys in results_dict.copy().keys():
        if target_var not in keys:
            del results_dict[keys]
            
    return results_dict

def print_dictionary_items(a_dict):

    for keys, items in a_dict.items():
        print()
        print(keys)
        print(items)

if __name__ == '__main__':
    a_matrix()

def rel_plots(df, target_var, x_list, delineation):
    for an_x in x_list:
        p1 = sns.relplot(kind='scatter', x=an_x, y=target_var, data=df, col=delineation,
        height=3.5, aspect=1)
        p1.map_dataframe(sns.lineplot, an_x, target_var, color='orange')

def basic_feature_plots(df):
    """Plots the basic features (bedroomcnt, bathroom cnt, square
    feet, and yearbuilt) against the target variable taxvaluedollarcnt.
    First two plots are scatter plots with a line plot added showing the 
    mean value of taxvaluedollarcnt for that number of value. Second two
    are regplots that have a regression line plotted."""
    fig, axes = plt.subplots(2, 2, figsize=(10,8))

    p1 = sns.scatterplot(x='bedroomcnt', y='taxvaluedollarcnt', 
                     data=df, ax=axes[0,0])
    p2 = sns.lineplot(data=df, x='bedroomcnt', y='taxvaluedollarcnt',
                      color='orange', ax=axes[0,0])

    p3 = sns.scatterplot(x='bathroomcnt', y='taxvaluedollarcnt', 
                     data=df, ax=axes[0,1])
    p4 = sns.lineplot(data=df, x='bathroomcnt', y='taxvaluedollarcnt',
                      color='orange', ax=axes[0,1])

    p4 = sns.regplot(data=df,x='sqr_ft',y='taxvaluedollarcnt', fit_reg=True,
                    line_kws={'color': 'orange'}, ax=axes[1,0])
    p5 = sns.regplot(data=df,x='yearbuilt',y='taxvaluedollarcnt', fit_reg=True,
                    line_kws={'color': 'orange'}, ax=axes[1,1])

def graph_list_of_clusters(clusters, different_plots=False):


    colors = ['b', 'g', 'r', 'y', 'purple', 'orange']
    clusters = clusters

    if different_plots:
        diff_plots(clusters, colors)
    else:
        plot_all(clusters, colors)


def plot_all(clusters, colors):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, cluster in enumerate(clusters):
        x, y, z = cluster.latitude, cluster.longitude, cluster.elevation
        ax.scatter(x,y, marker='s',c=colors[i], s=20, alpha=.6)
    plt.show()

def diff_plots(clusters, colors):

    for i, cluster in enumerate(clusters):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y, z = cluster.latitude, cluster.longitude, cluster.elevation
        ax.scatter(x,y, marker='s',c=colors[i], s=20, alpha=.6)
        plt.show()

def return_inverse_scaled_mean(scaler1, target, train, df):
    target_index = target_col_index(target, train)
    scaler2 = MinMaxScaler()
    scaler2.min_, scaler2.scale_ = scaler1.min_[target_index], scaler1.scale_[target_index]

    return scaler2.inverse_transform(df[target].values.reshape(-1,1))


def target_col_index(target, train):
    for i, col in enumerate(train.columns):
        if col == target:
            return i

    
