from module import transforms as trans
import numpy as np

model = trans.Transformer([
    trans.LeakyReLU(),
    trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    trans.LogRescale(), trans.RNNAffineCoupling()
])

tx = np.arange(30).reshape(5, 6).astype(np.float32) - 15
res = model(tx)
print(res)
print(model.losses)
print(sum(model.losses))

rev = model(res, training=False)
print(rev)
print(model.losses)
print(sum(model.losses))
