# from boilerplate.features import make_vectors
# from boilerplate.loader import transform_conll_to_vectors
# from boilerplate.mentions_custom import increment_mention, increment_mention_pair

import os

from tqdm import tqdm

from boilerplate import mock_trainer
from boilerplate import saver
from boilerplate.features import make_vectors
from boilerplate.loader import transform_conll_to_vectors
from boilerplate.mentions_custom import increment_mention, increment_mention_pair

root = r'F:\Users\loliveira\Documents\ProjetoFinal\dataset'
data_root = os.path.join(root, "conll-formatted-ontonotes-5.0-master\conll-formatted-ontonotes-5.0\data\development")

# This will create the files as vectors
# methods increment_mention, increment_mention_pair, make_vectors should be
# replaced in real implementation
transform_conll_to_vectors(data_root, r"{}\as_vectors".format(root), increment_mention, increment_mention_pair,
                           make_vectors)

# This will read the files and create a mock implementation. Should be replaced by a real algorithm
for f in tqdm(os.listdir(r"{}\as_vectors".format(root))):
    if f.endswith("_in"):
        f_in = f
        f_out = f[:-3] + "_out"
        doc = mock_trainer.predict(os.path.join(root, "as_vectors", f_in), os.path.join(root, "as_vectors", f_out))
        saver.save_document(os.path.join(data_root, r"data\english\annotations"), os.path.join(root, "output"), doc)
