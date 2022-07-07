import pandas as pd
from pathlib import Path

class Data:
    def get_text_list_from_files(paths):
        text_list = []
        for name in tqdm(paths):
            with open(name, mode='r') as f:
                text_list.append(f.read())
            f.close()
        return text_list

    def get_data_from_text_files(self,paths):

        texts = self.get_text_list_from_files(paths)
        df = pd.DataFrame(
            {
                "text": texts
            }
        )
        df = df.sample(len(df)).reset_index(drop=True)
        return df


