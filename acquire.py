from isort import file
import pandas as pd
import numpy as np
import os
from env import get_db_url

"""Make a function named get_titanic_data 
that returns the titanic data from the codeup data
science database as a pandas data frame. Obtain 
your data from the Codeup Data Science Database
"""

def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_db_url('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  

"""Make a function named get_iris_data that returns
the data from the iris_db on the codeup data 
science database as a pandas data frame. The 
returned data frame should include the actual 
name of the species in addition to the species_ids.
Obtain your data from the Codeup Data Science 
Database.
"""

def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        sql = """
        SELECT *
        FROM measurements
        JOIN species USING(species_id)
        """
        df = pd.read_sql(sql, get_db_url('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df 

"""Make a function named get_telco_data that 
returns the data from the telco_churn database in 
SQL. In your SQL, be sure to join all 4 tables 
together, so that the resulting dataframe 
contains all the contract, payment, and internet 
service options. Obtain your data from the Codeup 
Data Science Database.
"""

def get_telco_data():
    filename = "telco.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        sql = """
        SELECT *
        FROM customers
        JOIN internet_service_types USING(internet_service_type_id)
        JOIN payment_types USING(payment_type_id)
        JOIN contract_types USING(contract_type_id)
        """
        df = pd.read_sql(sql, get_db_url('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df 

"""
Once you've got your get_titanic_data, 
get_iris_data, and get_telco_data functions 
written, now it's time to add caching to them. 
To do this, edit the beginning of the function 
to check for the local filename of telco.csv, 
titanic.csv, or iris.csv. If they exist, use the 
.csv file. If the file doesn't exist, then 
produce the SQL and pandas necessary to create a 
dataframe, then write the dataframe to a .csv 
file with the appropriate name.
"""

def get_mall_customer_data():
    filename = 'mall_customer.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        sql = """
        SELECT *
        FROM customers;
        """

        df = pd.read_sql(sql, get_db_url('mall_customers'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df


def get_zillow_data():
    filename = 'zillow.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        sql = """
        SELECT *
        FROM properties_2017
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN storytype USING(storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusedesc = 'Single Family Residential';

        """
        df = pd.read_sql(sql, get_db_url('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        print('the last part')
        # Return the dataframe to the calling code
        return df 

if __name__ == '__main__':
    titanic = get_titanic_data()
    iris = get_iris_data()
    telco = get_telco_data()

    print(titanic.head())
    print(iris.head())
    print(telco.head())