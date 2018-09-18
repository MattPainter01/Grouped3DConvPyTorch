import torch
import tensor_comprehensions as tc


class Conv3DTC(torch.nn.Module): # TODO: Implement bias
    def __init__(self, I, C, K, groups=1, padding=0, bias=False, from_cache=False, cache_file='tc_group3d.pt', tuner_config=None):
        '''
        Module providing grouped 3d convolution using tensor comprehensions

        :param I: Number of input channels
        :type I: int
        :param C: Number of output channels
        :type C: int
        :param K: Kernel size
        :type K: tuple or int
        :param groups: Number of groups
        :type groups: int
        :param from_cache: If True load from specified cache file, If False, perform autotuning
        :type from_cache: bool
        :param cache_file: Path and name of cache file
        :type cache_file: string
        :param padding: Amount of input padding
        :type padding: tuple or int
        :param bias: Not implemented
        :type bias: bool
        :param tuner_config: Tuner config object to use for auto-tuning
        :type tuner_config: tensor_comprehensions.TunerConfig
        '''
        import torch.nn.functional as F
        super().__init__()

        K = self.int_to_tuple(K)
        padding = self.int_to_tuple(padding)

        group_convolution = self.tc_string()
        if not from_cache:
            if tuner_config is None:
                tuner_config = tc.TunerConfig().generations(25).pop_size(100).number_elites(15)
            TC = tc.define(group_convolution, tc.make_autotuned_options_factory(
                    tuner_config=tuner_config,
                    cache_filename=cache_file,
                    store_to_cache=True,
                    load_from_cache=True
                    ))
        else:
            TC = tc.define(group_convolution, tc.make_load_from_cache_options_factory(cache_file))

        self.convolution_grouped = tc.make_autograd(TC.group_convolution, TC.convolution_grad)
        self.W = torch.nn.Parameter(torch.rand(groups, C/groups, I/groups, K[0], K[1], K[2]))
        self.pad = F.pad
        self.groups = groups
        self.padding = padding
        self.K = K

    def int_to_tuple(self, val):
        if isinstance(val, int):
            return (val, val,val)
        else:
            return val

    '''
        Group convolution string.
        Backward pass based on code provided by github user kevjshih here: 
        https://github.com/facebookresearch/TensorComprehensions/issues/605
    '''
    def tc_string(self):
        group_convolution = """
        def group_convolution(float(N,G,C,T,H,W) I, float(G,F,C,KT,KH,KW) W1) -> (O)
        {
            O(n, g, f, t, h, w) +=! I(n, g, c, t + r_kt, h + r_kh, w + r_kw) * W1(g, f, c, r_kt, r_kh, r_kw) 
        }
        def convolution_grad(float(N,G,C,T,H,W) I, float(G,M,C,KT,KH,KW) W1, float(N,G,M,OT,OH,OW) d_O) -> (d_I, d_W1) {
            d_I(n, g, c, t, h, w) +=! ((t - r_kt >= 0) && (h - r_kh >= 0) && (w - r_kw >= 0) && (t - r_kt + KT <= T) && (w - r_kw + KW <= W) && (h - r_kh +KH <= H)) ? d_O( n, g, r_m, t - r_kt, h - r_kh, w - r_kw) * W1(g, r_m, c, r_kt, r_kh, r_kw) : 0 where r_kt in 0:KT, r_kh in 0:KH, r_kw in 0:KW, t in 0:T, h in 0:H, w in 0:W, r_m in 0:M, n in 0:N, c in 0:C
            d_W1(g, m, c, kt, kh, kw) +=! ((r_t >= kt) && (r_h >= kh) && (r_w >=kw) && (r_t - kt + KT <= T) && (r_w - kw + KW <= W) && (r_h - kh +KH <= H) ) ? d_O( r_n, g, m, r_t-kt, r_h-kh, r_w-kw) * I(r_n, g, c, r_t, r_h, r_w) : 0 where kt in 0:KT, kh in 0:KH, kw in 0:KW, r_t in 0:T, r_h in 0:H, r_w in 0:W, m in 0:M, c in 0:C
        }
        """
        return group_convolution

    def forward(self, x):
        x = x.view(x.shape[0], self.groups, x.shape[1]/self.groups, x.shape[-3], x.shape[-2], x.shape[-1])
        if self.padding[0] > 0 or self.padding[1] > 0 or self.padding[2] > 0:
            t, h, w = self.padding[0]//2, self.padding[1]//2, self.padding[2]//2
            x = self.pad(x, (w,w,h,h,t,t))
        x = self.convolution_grouped(x, self.W)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[-3], x.shape[-2], x.shape[-1])
        return x

