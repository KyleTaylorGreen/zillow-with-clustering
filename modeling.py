from appscript import k
from cv2 import KMEANS_PP_CENTERS
from sklearn.cluster import KMeans


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import wrangle
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from geopy import distance
from sklearn.feature_selection import SelectKBest, f_regression
import folium 

def recursive_feature_elim(x_train_scaled, y_train):
    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=2)

    # fit the data using RFE
    rfe.fit(x_train_scaled, y_train)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = x_train_scaled.iloc[:,feature_mask].columns.tolist()

def create_and_fit_rfe(x_train_scaled, y_train,
                       features_to_select =2):
    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=features_to_select)
    
    # fit rfe 
    rfe.fit(x_train_scaled, y_train)

    return rfe

def k_best(x_train_scaled, y_train):
    k_features = []
    #print(x_train_scaled.columns)

    for i in range(1, len(x_train_scaled.columns)+1):
        f_selector = SelectKBest(f_regression, k=i)
        f_selector.fit(x_train_scaled, y_train)

        feature_mask = f_selector.get_support()
        f_feature = x_train_scaled.iloc[:,feature_mask].columns.tolist()
        k_features.append(f_feature)

    return k_features


def rfe_ranking(rfe, x_train_scaled):
        # view list of columns and their ranking

        # get the ranks
        var_ranks = rfe.ranking_
        # get the variable names
        var_names = x_train_scaled.columns.tolist()
        # combine ranks and names into a df for clean viewing
        rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})

        return rfe_ranks_df

def y_mean_median_base_pred(y_train, y_validate, target_var):
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # TVDC -> taxvaluedollarcnt
    TVDC_pred_mean = y_train[target_var].mean()
    y_train['TVDC_pred_mean'] = TVDC_pred_mean
    y_validate['TVDC_pred_mean'] = TVDC_pred_mean

    TVDC_pred_median = y_train[target_var].median()
    y_train['TVDC_pred_median'] = TVDC_pred_median
    y_validate['TVDC_pred_median'] = TVDC_pred_median

    return y_train, y_validate

def base_RMSE(y_train, y_validate):
    
    # rmse of baseline mean
    rmse_train_mean_bl = mean_squared_error(y_train.logerror, y_train.TVDC_pred_mean)**(1/2)
    rmse_val_mean_bl = mean_squared_error(y_validate.logerror, y_validate.TVDC_pred_mean)**(1/2)

    # rmse of baseline median
    rmse_train_med_bl = mean_squared_error(y_train.logerror, y_train.TVDC_pred_median)**(1/2)
    rmse_val_med_bl = mean_squared_error(y_validate.logerror, y_validate.TVDC_pred_median)**(1/2)

    return rmse_train_mean_bl, rmse_val_mean_bl,\
           rmse_train_med_bl, rmse_val_med_bl


class LRM:
    """Class object to hold many descriptive values and train, validate
    test splits for zillow data, as well as scaled and polynomial transformations
    of each. Also includes methods to use the previously mentioned data for 
    the type of regression requested."""

    def __init__(self, df, target_var, county='all'):
        """Initialize many different values and set splits for linear regression"""
        # splits input df into train, validate, test
        # splits by county if county is specified
        self.county = county

        # creates empty record base for each type of regression
        self.tweedies = pd.DataFrame()
        self.OLS = pd.DataFrame()
        self.lassolars = pd.DataFrame()
        self.poly = pd.DataFrame()
        
        # maintains target variable
        self.target_var = target_var

        # create train, validate and test splits,
        # as well as scaling and appending scaled data to 
        # train, validate and test splits.
        self.train, self.validate, self.test, \
        self.x_train_scaled, self.y_train, \
        self.x_validate_scaled, self.y_validate, \
        self.x_test_scaled, self.y_test = wrangle.all_train_validate_test_data(df, target_var, county)
        
        # polynomial transform train, validate & test features 
        # with degree of 2
        self.x_train_degree2,\
        self.x_validate_degree2,\
        self.x_test_degree2 = self.polynomial(degree=2)


        # polynomial transform train, validate & test features 
        # with degree of 2
        self.x_train_degree3,\
        self.x_validate_degree3,\
        self.x_test_degree3 = self.polynomial(degree=3)


        self.features_to_select= len(self.x_train_scaled.columns)
        self.rfe = create_and_fit_rfe(self.x_train_scaled,
                                      self.y_train,
                                      self.features_to_select)

        self.rfe_mask = self.rfe.support_
        self.rfe_features = self.x_train_scaled.iloc[:,self.rfe.support_].columns.tolist()
        self.rfe_ranks = rfe_ranking(self.rfe, self.x_train_scaled)
        self.rfe_feat_count = len(self.rfe_features)
        
        self.y_train, self.y_validate = y_mean_median_base_pred(self.y_train,
                                                                self.y_validate,
                                                                target_var)
        self.rmse_train_mean_bl,\
        self.rmse_val_mean_bl, \
        self.rmse_train_med_bl,\
        self.rmse_val_med_bl = base_RMSE(self.y_train, self.y_validate)

        self.normalized_rmse_train = round(self.rmse_train_mean_bl/self.train.logerror.mean(), 2)
        self.normalized_rmse_val = round(self.rmse_val_mean_bl/self.validate.logerror.mean(), 2)

    def OLS_regression(self, use_rfe_features=False):
        """Fits an Ordinary Least Squares Regression to the data set
        and makes predictions on train and validate sets. Returns stats
        of the model such as the name, the RSME on train and validate, and a
        normalized RMSE to serve as another quick indicator of performance."""
        
        model_name = 'OLS'

        if use_rfe_features:
            rfe_features = self.rfe_features
            x_train = self.x_train_scaled[rfe_features]
            x_validate = self.x_validate_scaled[rfe_features]

        else:
            x_train = self.x_train_scaled
            x_validate = self.x_validate_scaled

        #print(x_train)

        # create the model object
        lm = LinearRegression(normalize=True)

        # fit the model to our training data. We must specify the column in self.y_train, 
        # since we have converted it to a dataframe from a series! 
        lm.fit(x_train, self.y_train.logerror)

        # predict train
        self.y_train['TVDC_pred_OLS'] = lm.predict(x_train)

        # evaluate: rmse
        rmse_train = mean_squared_error(self.y_train.logerror, self.y_train.TVDC_pred_OLS)**(1/2)

        # predict validate
        self.y_validate['TVDC_pred_OLS'] = lm.predict(x_validate)

        # evaluate: rmse
        rmse_validate = mean_squared_error(self.y_validate.logerror, self.y_validate.TVDC_pred_OLS)**(1/2)

        model_stats = self.make_model_stats(model_name, rmse_train, rmse_validate,
                              x_train)
        self.OLS = pd.concat([self.OLS, model_stats], ignore_index=True)

        return model_stats

    def loop_OLS_regression(self):
        list_of_kbest = k_best(self.x_train_scaled, self.y_train.logerror)
        #print(list_of_kbest)
        for kbest in list_of_kbest:
            #print(kbest)
            self.rfe_features = kbest
            self.OLS_regression(use_rfe_features=True)

    def poly_regression(self, use_rfe_features=False, degree=2):
        """fits the features to a polynomial transformation and then 
        fits a regression based on those transformed features. Returns a
        descriptive df of the model, also is concated to a record of all
        poly regressions used."""

        model_name = 'polynomial'
        
        if use_rfe_features:
            rfe_features = self.rfe_features
            #print(self.x_validate_degree2)
            if degree==2:
                x_train = self.x_train_degree2[rfe_features]
                x_validate = self.x_validate_degree2[rfe_features]
            if degree==3:
                x_train = self.x_train_degree3[rfe_features]
                x_validate = self.x_validate_degree3[rfe_features]
        else:
            if degree==2:
                x_train = self.x_train_degree2
                x_validate = self.x_validate_degree2
            if degree==3:
                x_train = self.x_train_degree3
                x_validate = self.x_validate_degree3

        #print(x_train)

        # create the model object
        lm = LinearRegression(normalize=True)

        # fit the model to our training data. We must specify the column in self.y_train, 
        # since we have converted it to a dataframe from a series! 
        lm.fit(x_train, self.y_train.logerror)

        # predict train
        self.y_train['TVDC_pred_poly'] = lm.predict(x_train)

        # evaluate: rmse
        rmse_train = mean_squared_error(self.y_train.logerror, self.y_train.TVDC_pred_poly)**(1/2)

        # predict validate
        self.y_validate['TVDC_pred_poly'] = lm.predict(x_validate)

        # evaluate: rmse
        rmse_validate = mean_squared_error(self.y_validate.logerror, self.y_validate.TVDC_pred_poly)**(1/2)

        model_stats = self.make_model_stats(model_name, rmse_train, rmse_validate,
                              x_train)
        self.poly = pd.concat([self.poly, model_stats], ignore_index=True)

        return model_stats


    def polynomial(self, degree):
        # make the polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=degree)

        # fit and transform X_train_scaled
        #print(self.x_train_scaled)
        x_train_degree2 = pf.fit_transform(self.x_train_scaled)
        #print(x_train_degree2)

        # transform X_validate_scaled & X_test_scaled
        x_validate_degree2 = pf.transform(self.x_validate_scaled)
        x_test_degree2 = pf.transform(self.x_test_scaled)

        return x_train_degree2, x_validate_degree2, x_test_degree2


    def lassolars_regression(self, alpha=1, use_rfe_features=False):
        """Performs a lassolars regression by fitting training data and predicitng
        train and validate sets. Results are returned in a df and also appended to a 
        record of all other lassolars_regressions."""

        model_name = 'lasso_lars'
        rfe_feat = self.rfe_features
        if use_rfe_features:
            x_train = self.x_train_scaled[rfe_feat]
            x_validate = self.x_validate_scaled[rfe_feat]

        else:
            x_train = self.x_train_scaled
            x_validate = self.x_validate_scaled

        # create the model object
        lars = LassoLars(alpha=alpha)

        # print(x_train)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lars.fit(x_train, self.y_train.logerror)

        # predict train
        self.y_train['TVDC_pred_lars'] = lars.predict(x_train)

        # evaluate: rmse
        rmse_train = mean_squared_error(self.y_train.logerror, self.y_train.TVDC_pred_lars)**(1/2)

        # predict validate
        self.y_validate['TVDC_pred_lars'] = lars.predict(x_validate)

        # evaluate: rmse
        rmse_validate = mean_squared_error(self.y_validate.logerror, self.y_validate.TVDC_pred_lars)**(1/2)

        model_stats = self.make_model_stats(model_name, rmse_train, rmse_validate,
                              x_train, alpha=alpha)
        self.lassolars = pd.concat([self.lassolars, model_stats], ignore_index=True)

        return model_stats
    
    
    def tweedie(self, power=1, alpha=0, use_rfe_features=False):
        """Performs a tweedie regression by fitting training data and predicitng
        train and validate sets. Results are returned in a df and also appended to a 
        record of all other tweedie regressions."""

        model_name = 'tweedie'
        rfe_feat = self.rfe_features
        if use_rfe_features:
            x_train = self.x_train_scaled[rfe_feat]
            x_validate = self.x_validate_scaled[rfe_feat]

        else:
            x_train = self.x_train_scaled
            x_validate = self.x_validate_scaled
            # create the model object
        glm = TweedieRegressor(power=1, alpha=0)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        glm.fit(x_train, self.y_train.logerror)

        # predict train
        self.y_train['TVDC_pred_glm'] = glm.predict(x_train)

        # evaluate: rmse
        rmse_train = mean_squared_error(self.y_train.logerror, self.y_train.TVDC_pred_glm)**(1/2)

        # predict validate
        self.y_validate['TVDC_pred_glm'] = glm.predict(x_validate)

        # evaluate: rmse
        rmse_validate = mean_squared_error(self.y_validate.logerror, self.y_validate.TVDC_pred_glm)**(1/2)

        model_stats = self.make_model_stats(model_name, rmse_train, rmse_validate,
                              x_train, alpha=alpha, power=power)

        self.tweedies = pd.concat([self.tweedies, model_stats], ignore_index=True)

        return pd.DataFrame(model_stats)

    
    def all_models_df(self):
        df= pd.concat([self.OLS, self.lassolars, self.tweedies, self.poly], ignore_index=True)

        return df

    def make_model_stats(self, model_name, rmse_train, rmse_validate, x_train, alpha=0, power=0):
        model_stats = {}

        model_stats['model_name'] = [model_name]
        model_stats['county'] = [self.county]
        model_stats['rmse_train'] = [rmse_train]
        model_stats['rmse_validate'] = [rmse_validate]

        if model_name == 'OLS' or model_name == 'lasso_lars':
            model_stats['power'] = ['NA']
        else:
            model_stats['power'] = [power]
        
        if model_name == 'lasso_lars' or model_name == 'tweedie':
            model_stats['alpha'] = [alpha]
        else:
            model_stats['alpha'] = ['NA']
        
    
       # model_stats['features'] = [x_train.columns]

        model_stats = pd.DataFrame(model_stats)
        model_stats = self.percent_diff(model_stats)
        model_stats = self.baseline_diff(model_stats)

        return model_stats

    def percent_diff(self, model_stats):
        model_stats['percent_diff'] = round((model_stats['rmse_train'] - model_stats['rmse_validate'])/model_stats['rmse_train'] * 100, 2)

        return model_stats

    def baseline_diff(self, model_stats):
        model_stats['norm_rmse_train'] = round((model_stats['rmse_train']) / self.train.logerror.mean(), 4)
        model_stats['norm_rmse_validate'] = round((model_stats['rmse_validate']) / self.validate.logerror.mean(), 4)
        
        return model_stats

def make_map(train):   
    m = folium.Map(
        location = [train['latitude'].mean(), train['longitude'].mean()],
        zoom_start = 6)
    return m


def make_clusters(df, n_clusters, title, col_list=False, total_df=False):
    """ 
    Input a df and return a list of dataframes separated by cluster. 
    If total_df == True: return one whole df with all clusters.

    df = input dataframe, clusters will be made on features passed through
    OR the entire df if no columns passed through.
    
    n_clusters = number of clusters to use in KMeans

    col_list = list of columns to include as features for the cluster
    to evaluate.

    total_df = False by default, returns entire df with cluster group column
    added if True. If false, returns list of dfs subset of each cluster group.
    """
    coords = ['latitude', 'longitude', 'elevation']

    if col_list:
        X = df[col_list]
        final_df = pd.concat([df[col_list], df[coords]],axis=1)
    else:
        X = df

    # create KMeans object with n clusters    
    kmeans = KMeans(n_clusters=n_clusters)
    
    # fit kmeans clusters and then predict on features
    kmeans.fit(X)
    X[f'{title}_cluster_groups'] = kmeans.predict(X)
    final_df = pd.concat([final_df, X[f'{title}_cluster_groups']], axis=1)

    # returns entire df or makes list of dfs separated by cluster group
    if total_df:
        return final_df
    else:
        return separated_cluster_dfs(final_df, title)


def separated_cluster_dfs(cluster_df, title):
    """ 
    Takes a df with cluster groups and then returns a list of dfs 
    separated by cluster.

    cluster_df = the dataframe to be split by cluster group

    returns list of dfs separated by cluster.
    """
    list_of_dfs_by_cluster = []

    # for each cluster number, append the df indexed by each cluster 
    # to a list to be returned
    for cluster in cluster_df.cluster_groups.unique():
        list_of_dfs_by_cluster.append(cluster_df[cluster_df[f'{title}cluster_groups'] == cluster])
    
    return list_of_dfs_by_cluster


