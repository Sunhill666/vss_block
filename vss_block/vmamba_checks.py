import time
import math
import copy
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from .vmamba import CrossScan, CrossMerge, getCSM
from .vmamba import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
from .vmamba import VSSM, PatchMerging2D, Mlp, gMlp, LayerNorm2d, VSSBlock


class CHECKS:
    def check_vssm_equals_vmambadp():
        try:
            from _ignore.vmamba.vmamba_bak1 import VMamba2Dp
            from _ignore.vmamba.vmamba_pub import VSSM
        except Exception:
            print("original VSSM and VMamba2Dp not found.", flush=True)
            return

        # test 1 True =================================
        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())
        oldvss = VMamba2Dp(depths=[2, 2, 6, 2]).half().cuda()
        newvss = VSSM(depths=[2, 2, 6, 2]).half().cuda()
        newvss.load_state_dict(oldvss.state_dict())
        input = torch.randn((12, 3, 224, 224)).half().cuda()
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward_backbone(input)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward_backbone(input)
        # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        print((y1 - y2).abs().sum())

        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward(input)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward(input)
        # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        print((y1 - y2).abs().sum())

        # test 2 True ==========================================
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        oldvss = VMamba2Dp(depths=[2, 2, 6, 2]).cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        newvss = VSSM(depths=[2, 2, 6, 2]).cuda()

        miss_align = 0
        for k, v in oldvss.state_dict().items():
            same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
            if not same:
                print(k, same)
                miss_align += 1
        print("init miss align", miss_align)  # init miss align 0

    def check_vssm1_equals_vssm(forward_type="v0"):
        try:
            from _ignore.vmamba.vmamba_pub import VSSM as VSSM0
        except Exception:
            print("original VSSM and VMamba2Dp not found.", flush=True)
            return

        class VSSM_(VSSM):
            @staticmethod
            def _make_layer(*args, **kwargs):
                layer = VSSM._make_layer(*args, **kwargs)
                dim = kwargs.get("dim", None)
                norm_layer = kwargs.get("norm_layer", None)
                downsample = kwargs.get("downsample", None)
                blocks = layer.blocks

                if True:  # is this really applied? Yes, but been overriden later in VSSM!
                    def _init_weights(module: nn.Module):
                        for name, p in module.named_parameters():
                            if name in ["out_proj.weight"]:
                                # fake init, just to keep the seed ....
                                p = p.clone().detach_()
                                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    blks = nn.Sequential(*copy.deepcopy(blocks))
                    blks.apply(_init_weights)

                downsample = PatchMerging2D(
                    dim, 2*dim, norm_layer=norm_layer) if downsample is None else nn.Identity()

                return nn.Sequential(OrderedDict(
                    blocks=nn.Sequential(*blocks,),
                    downsample=downsample,
                ))

            def forward_backbone(self, x):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                return x

            def forward1(self, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = self.classifier.norm(x)
                # here: whether has contiguous would differ
                x = self.classifier.avgpool(
                    x.permute(0, 3, 1, 2).contiguous()).flatten(1)
                x = self.classifier.head(x)
                return x

        # only has initial difference
        VSSM1 = partial(VSSM, downsample_version="v1", patchembed_version="v1",
                        mlp_ratio=0.0, ssm_ratio=2.0, forward_type=forward_type)
        VSSM.forward_backbone = VSSM_.forward_backbone
        VSSM.forward1 = VSSM_.forward1
        # expected to be all the same
        VSSM1 = partial(VSSM_, downsample_version="none", patchembed_version="v1",
                        mlp_ratio=0.0, ssm_ratio=2.0, forward_type=forward_type)

        # test 1 True =================================
        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())
        oldvss = VSSM0(depths=[2, 2, 6, 2]).half().cuda()
        newvss = VSSM1(depths=[2, 2, 6, 2]).half().cuda()
        newvss.load_state_dict(oldvss.state_dict())
        input = torch.randn((12, 3, 224, 224)).half().cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward_backbone(input)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward_backbone(input)
        # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        print((y1 - y2).abs().sum())

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward(input)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward1(input)
        # tensor(2.5988e-05, device='cuda:0', grad_fn=<SumBackward0>)
        print((y1 - y2).abs().sum())
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y3 = newvss.forward(input)
        # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        print((y1 - y3).abs().sum())

        # test 2 True ==========================================
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        oldvss = VSSM0(depths=[2, 2, 6, 2]).cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        newvss = VSSM1(depths=[2, 2, 6, 2]).cuda()

        miss_align = 0
        oldvss2new = copy.deepcopy(newvss)
        oldvss2new.load_state_dict(oldvss.state_dict())
        for k, v in oldvss2new.state_dict().items():
            same = (oldvss2new.state_dict()[k] == newvss.state_dict()[k]).all()
            if not same:
                print(k, same)
                miss_align += 1
        print("init miss align", miss_align)  # init miss align 0

    def check_vssm1_ssoflex_equals_mambassm():
        # only has initial difference
        VSSM0 = partial(VSSM, downsample_version="v3", patchembed_version="v2",
                        mlp_ratio=4.0, ssm_ratio=2.0, forward_type="v2")
        VSSM1 = partial(VSSM, downsample_version="v3", patchembed_version="v2",
                        mlp_ratio=4.0, ssm_ratio=2.0, forward_type="v01")

        # test 1 True =================================
        torch.manual_seed(time.time())
        torch.cuda.manual_seed(time.time())
        oldvss = VSSM0(depths=[2, 2, 6, 2]).half().cuda()
        newvss = VSSM1(depths=[2, 2, 6, 2]).half().cuda()
        newvss.load_state_dict(oldvss.state_dict())
        input0 = torch.randn((12, 3, 224, 224)).half().cuda().requires_grad_()
        input1 = input0.detach().clone().requires_grad_()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward(input0)
            y1.sum().backward()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward(input1)
            y2.sum().backward()
        # tensor(0., device='cuda:0', dtype=torch.float16, grad_fn=<SumBackward0>)
        print((y1 - y2).abs().sum())
        # tensor(6.6016, device='cuda:0', dtype=torch.float16)
        print((input0.grad - input1.grad).abs().sum())

        # test 2 True ==========================================
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        oldvss = VSSM0(depths=[2, 2, 6, 2]).cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        newvss = VSSM1(depths=[2, 2, 6, 2]).cuda()

        miss_align = 0
        oldvss2new = copy.deepcopy(newvss)
        oldvss2new.load_state_dict(oldvss.state_dict())
        for k, v in oldvss2new.state_dict().items():
            same = (oldvss2new.state_dict()[k] == newvss.state_dict()[k]).all()
            if not same:
                print(k, same)
                miss_align += 1
        print("init miss align", miss_align)  # init miss align 0

    def check_csm_triton():

        B, C, H, W = 256, 192, 56, 57
        dtype = torch.float16
        dtype = torch.float32
        x = torch.randn((B, C, H, W), dtype=dtype,
                        device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 4, C, H, W), dtype=dtype,
                        device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x.view(B, C, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x.contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(
                    x, dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs

        def cross_merge(out_y: torch.Tensor):
            B, K, D, H, W = out_y.shape
            L = H * W
            out_y = out_y.view(B, K, D, L)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(
                B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(
                B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            return y

        def cross_scan_1b1(x: torch.Tensor):
            B, K, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x[:, 0].view(B, C, L),
                torch.transpose(x[:, 1], dim0=2,
                                dim1=3).contiguous().view(B, C, L),
                torch.flip(x[:, 2].contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(
                    x[:, 3], dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs

        def unidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
            return x

        def unidi_merge(ys):
            B, K, C, H, W = ys.shape
            return ys.view(B, 4, -1, H * W).sum(1)

        def bidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            x = torch.cat([x, x.flip(dims=[-1])], dim=1)
            return x

        def bidi_merge(ys):
            B, K, D, H, W = ys.shape
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            return ys.contiguous().sum(1)

        if True:
            res0 = triton.testing.do_bench(lambda: cross_scan(x))
            res1 = triton.testing.do_bench(lambda: CrossScan.apply(x))
            res2 = triton.testing.do_bench(lambda: CrossScanTriton.apply(x))
            res3 = triton.testing.do_bench(lambda: cross_merge(y))
            res4 = triton.testing.do_bench(lambda: CrossMerge.apply(y))
            res5 = triton.testing.do_bench(lambda: CrossMergeTriton.apply(y))
            print(res0, res1, res2, res3, res4, res5)
            res0 = triton.testing.do_bench(
                lambda: cross_scan(x).sum().backward())
            res1 = triton.testing.do_bench(
                lambda: CrossScan.apply(x).sum().backward())
            res2 = triton.testing.do_bench(
                lambda: CrossScanTriton.apply(x).sum().backward())
            res3 = triton.testing.do_bench(
                lambda: cross_merge(y).sum().backward())
            res4 = triton.testing.do_bench(
                lambda: CrossMerge.apply(y).sum().backward())
            res5 = triton.testing.do_bench(
                lambda: CrossMergeTriton.apply(y).sum().backward())
            print(res0, res1, res2, res3, res4, res5)

        print("test cross scan")
        for (cs0, cm0, cs1, cm1) in [
            (cross_scan, cross_merge, CrossScanTriton, CrossMergeTriton),
            (unidi_scan, unidi_merge, getCSM(1)[0], getCSM(1)[1]),
            (bidi_scan, bidi_merge, getCSM(2)[0], getCSM(2)[1]),
        ]:
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cs0(x)
            o1 = cs1.apply(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cm0(y)
            o1 = cm1.apply(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None

        print("test cross scan one by one")
        if True:
            o0 = cross_scan_1b1(y)
            o1 = CrossScanTriton1b1.apply(y1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None

        x = torch.arange(0, 16, 1).view(1, 1, 4, 4)
        print(x)
        print(cross_scan(x))
        print(cross_merge(cross_scan(x).view(1, 4, 1, 4, 4)))
        x = torch.arange(0, 16, 1).view(1, 4, 1, 2, 2)
        print(x)
        print(cross_scan_1b1(x))

    def check_einsum():
        B, D, L, R, K = 128, 192, 56 * 56, 12, 4
        o = torch.randn((B, K * D, L)).cuda()
        x = torch.randn((B, K, R, L)).cuda().requires_grad_(True)
        w = torch.randn((K, D, R)).cuda().requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        w1 = w.clone().detach().requires_grad_(True)

        y1 = torch.einsum("bkrl,kdr->bkdl", x, w).contiguous().view(B, -1, L)
        y2 = F.conv1d(x1.view(B, -1, L), w1.view(K * D, R, 1),
                      None, groups=K).contiguous().view(B, -1, L)
        print((y1 - y2).abs().max())
        y1.backward(o)
        y2.backward(o)
        print((x.grad - x1.grad).abs().max())

    def check_vssblock():
        import triton
        from torchvision.models.vision_transformer import EncoderBlock

        vb = VSSBlock(
            hidden_dim=16,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            ssm_d_state=1,
            ssm_ratio=2,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            use_checkpoint=False,
        ).cuda()

        trans = EncoderBlock(
            num_heads=1,
            hidden_dim=16,
            mlp_dim=int(4.0 * 16),
            dropout=0.0,
            attention_dropout=0.0,
            norm_layer=nn.LayerNorm,
        ).cuda()

        inp = torch.randn((16, 128, 128, 16)).cuda().requires_grad_()
        inp2 = inp.detach().cuda().view(16, -1, 16).requires_grad_()
        def fn(): return vb(inp)
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        def fn(): return trans(inp2)
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        def fn(): return vb(inp).sum().backward()
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        def fn(): return trans(inp2).sum().backward()
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        import time
        time.sleep(10000)

    def check_ln2d():
        import triton
        B, C, H, W = 128, 8192, 7, 7
        inp = torch.randn((B, C, H, W)).cuda().requires_grad_()
        inp2 = inp.detach().permute(0, 2, 3, 1).clone().requires_grad_()

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n1 = LayerNorm2d(C).cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n2 = nn.LayerNorm(C).cuda()
        o1 = n1(inp)
        o2 = n2(inp2)
        print((o1.permute(0, 2, 3, 1) - o2).abs().max())
        o1.backward(inp.data)
        o2.backward(inp.data.permute(0, 2, 3, 1))
        print((inp.grad.permute(0, 2, 3, 1) - inp2.grad).abs().max())

        ms1 = triton.testing.do_bench(lambda: n1(inp))
        ms2 = triton.testing.do_bench(lambda: n2(inp2))
        ms3 = triton.testing.do_bench(lambda: n1(inp))
        print(ms1, ms2, ms3)

    def check_linear_2d():
        import triton
        inp = torch.randn((64, 192, 56, 57)).cuda().requires_grad_()
        inp2 = inp.detach().permute(0, 2, 3, 1).clone().requires_grad_()

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n1 = Mlp(192, 4*192, 384, channels_first=True).cuda()
        catch_random1 = torch.randn((1,))
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n2 = Mlp(192, 4*192, 384, channels_first=False).cuda()
        catch_random2 = torch.randn((1,))
        print(catch_random1, catch_random2)
        with torch.cuda.amp.autocast():
            o1 = n1(inp)
            o2 = n2(inp2)
        print((o1.permute(0, 2, 3, 1) - o2).abs().max())
        o1.sum().backward()
        o2.sum().backward()
        print((inp.grad.permute(0, 2, 3, 1) - inp2.grad).abs().max())

        i1, i2 = inp.float(), inp2.float()
        ms2 = triton.testing.do_bench(lambda: n2(i2))
        ms1 = triton.testing.do_bench(lambda: n1(i1))
        ms4 = triton.testing.do_bench(lambda: n2(i2).sum().backward())
        ms3 = triton.testing.do_bench(lambda: n1(i1).sum().backward())
        print(ms1, ms2, ms3, ms4)

    def check_gmlp():
        import triton
        inp = torch.randn((64, 192, 56, 57)).cuda().requires_grad_()
        inp2 = inp.detach().permute(0, 2, 3, 1).clone().requires_grad_()

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n1 = Mlp(192, 4*192, 384, channels_first=True).cuda()
        catch_random1 = torch.randn((1,))
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n2 = gMlp(192, 2*192, 384, channels_first=False).cuda()
        catch_random2 = torch.randn((1,))
        print(catch_random1, catch_random2)
        with torch.cuda.amp.autocast():
            o1 = n1(inp)
            o2 = n2(inp2)
        print((o1.permute(0, 2, 3, 1) - o2).abs().max())
        o1.sum().backward()
        o2.sum().backward()
        print((inp.grad.permute(0, 2, 3, 1) - inp2.grad).abs().max())

        i1, i2 = inp.float(), inp2.float()
        ms2 = triton.testing.do_bench(lambda: n2(i2))
        ms1 = triton.testing.do_bench(lambda: n1(i1))
        ms4 = triton.testing.do_bench(lambda: n2(i2).sum().backward())
        ms3 = triton.testing.do_bench(lambda: n1(i1).sum().backward())
        print(ms1, ms2, ms3, ms4)

    def check_channel_first():
        import triton
        inp = torch.randn((64, 3, 224, 224)).cuda().half().requires_grad_()
        inp2 = inp.detach().clone().requires_grad_()

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n1 = VSSM(norm_layer="ln").cuda()
        catch_random1 = torch.randn((1,))
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        n2 = VSSM(norm_layer="ln2d").cuda()
        catch_random2 = torch.randn((1,))
        print(catch_random1, catch_random2)
        with torch.cuda.amp.autocast():
            o1 = n1(inp)
            o2 = n2(inp2)
        print((o1 - o2).abs().max())
        o1.sum().backward()
        o2.sum().backward()
        print((inp.grad - inp2.grad).abs().max())

        i1, i2 = inp.float(), inp2.float()
        ms2 = triton.testing.do_bench(lambda: n2(i2))
        ms1 = triton.testing.do_bench(lambda: n1(i1))
        ms4 = triton.testing.do_bench(lambda: n2(i2).sum().backward())
        ms3 = triton.testing.do_bench(lambda: n1(i1).sum().backward())
        print(ms1, ms2, ms3, ms4)

    def check_profile():
        torch.cuda.manual_seed(0)

        def trace_handler(prof: torch.profiler.profile):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # print(prof.export_chrome_trace("./tracev1.json"))

        with torch.cuda.amp.autocast():
            with torch.profiler.profile(
                with_modules=True,
                with_stack=True,
                profile_memory=True,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],

                # In this example with wait=1, warmup=1, active=2, repeat=1,
                # profiler will skip the first step/iteration,
                # start warming up on the second, record
                # the third and the forth iterations,
                # after which the trace will become available
                # and on_trace_ready (when set) is called;
                # the cycle repeats starting with the next step

                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=1),
                on_trace_ready=trace_handler
                # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
                # used when outputting for tensorboard
            ) as prof:
                for _ in range(1000):
                    # with torch.autograd.profiler.record_function("patch_embed"):
                    #     x = self.patch_embed(x)
                    prof.step()

    def load22kto1k():
        return


if __name__ == "__main__":
    # CHECKS.check_vssblock()
    # CHECKS.check_vssm_equals_vmambadp()
    # CHECKS.check_vssm1_equals_vssm(forward_type="v0")
    # CHECKS.check_vssm1_equals_vssm(forward_type="v0_seq")
    # CHECKS.check_vssm1_ssoflex_equals_mambassm()
    CHECKS.check_csm_triton()
    # CHECKS.check_einsum()
    # CHECKS.check_ln2d()
    # CHECKS.check_linear_2d()
    # CHECKS.check_gmlp()
    # CHECKS.check_channel_first()
    # breakpoint()
