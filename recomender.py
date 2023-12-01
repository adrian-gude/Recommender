import pandas as pd 
import matplotlib.pyplot as plt


def summarize_dataset_info(dataset):
    """
    Function to sumarize the dataset and visualize the info  

    Args:
        data_path ([type]): [string]

    """

    
    users = dataset['userId']
    rating = dataset['rating']
    products = dataset['movieId']

    missing_values = dataset.isna().sum()    
    duplicates = dataset.duplicated().sum()

    print("\nMising values :")
    print(missing_values)
    print("\nDuplicated values :", duplicates)

    print('Num users : ', users.nunique())
    print('Num ratings : ',rating.nunique())
    print('Num products : ',products.nunique())

    print("Dataset shape : ", dataset.shape)



def delete_items_from_dataset (dataset) : 
    """
    This function is used to remove items from the dataset. It takes a dataframe and removes all rows that has less than 10 products
    and users with less than 20 scores

    Args:
        dataset (pd.Dataset): 

    Returns:
        cleaned_df : cleaned dataset
    """

    product_counts = dataset['movieId'].value_counts()
    cleaned_df = dataset[dataset['movieId'].isin(product_counts[product_counts >= 10].index)]

    user_counts = cleaned_df['userId'].value_counts()
    cleaned_df = cleaned_df[cleaned_df['userId'].isin(user_counts[user_counts >= 20].index)]

    return cleaned_df 

def plot_info_dataset (dataset,column,title,xlabel,ylabel): 
    """
    plot an histogram of the dataset

    Args:
        dataset ([type]): [description]
        column ([type]): [description]
        title ([type]): [description]
        xlabel ([type]): [description]
        ylabel ([type]): [description]
    """
    user_column_count = dataset[column].value_counts()
    
    plt.bar(user_column_count.value_counts().index, user_column_count.value_counts().values, color='blue')
    plt.grid(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    data_path = "ml-latest-small/ratings.csv"

    raw_df = pd.read_csv(data_path)
    cleaned_df = delete_items_from_dataset(raw_df)

    summarize_dataset_info(raw_df)
    print("\n\nAFTER CLEANING\n\n")
    summarize_dataset_info(cleaned_df)

    print("cleaned_df[userId] ")
    print(cleaned_df['userId'])

    
    print(cleaned_df.head())
    
    # Ejercicicio 3 
    #plot_info_dataset(cleaned_df,'userId','ratings by users','ratings by users','num ratings by user')
    # plt.show()
    #plot_info_dataset(cleaned_df,'movieId','ratings by product','ratings by product','num ratings by product ')
    # plt.show()
    
    cleaned_df["userId"].value_counts().plot(kind='hist', title='Puntuaciones por usuario')
    plt.show()

    # raw_df["userId"].value_counts().plot(kind='hist', title='Puntuaciones por usuario')
    # plt.show()

    cleaned_df["movieId"].value_counts().plot(kind='hist', title='Puntuaciones por producto')
    plt.show()


    # Ejercicio 5 

    cleaned_df["rating"].plot(kind='hist', title='Puntuaciones por rating')
    plt.show()

    #plot_info_dataset(cleaned_df,'rating','Users by rating', 'rating', 'num users')
     


    