from data_split import DataSplitter
from dvc_operations.download import DVCReader

def split_data():
    """Split the data into train and test"""
    data_splitter = DataSplitter()
    dvc = DVCReader()

    data_path = 'data/credit_card_featured_engg.csv'
    dvc.read_csv(data_path)
    data_splitter.split_data(data_path)


if __name__ == "__main__":
    split_data()