"""Test softmax operators."""

# pylint: disable=missing-function-docstring, unused-import
import random
import pytest
import torch
import torch_musa
from torch_musa import testing

configs = [
    # shape, dim
    [(1024,), 0],
    [(4, 256), 1],
    [(4, 256, 2), 0],
    [(4, 256, 2, 2), 1],
    [(4, 1, 20, 20), 2],
    [(4, 20, 1, 1), 0],
    [(5, 64, 30, 2, 3), 1],
    [(7, 1, 2048, 2, 3, 4), 2],
    [(2, 4, 8, 16, 32, 2, 2), 3],
    [(2, 3, 4, 5, 6, 7, 8, 1), 2],
    [(0, 3, 4, 5, 6, 7, 8, 1), 2],
]


@pytest.mark.parametrize("columns", [511, 512, 513, 516, 537, 538, 539, 540, 1024])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_softmax_wide_row_boundaries(columns, noncontiguous):
    """Exercise the vector-width boundary, out variant, streams and autograd."""
    cpu = torch.sin(torch.arange(32 * columns).reshape(32, columns).float()) * 0.5
    cpu[:, columns // 2 :] = -10000.0
    if noncontiguous:
        cpu = cpu.t().contiguous().t()
    cpu.requires_grad_(True)
    x = cpu.detach().to("musa").requires_grad_(True)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        actual = torch.softmax(x, -1)
        out = torch.empty_like(actual)
        returned = torch.softmax(x.detach(), 1, out=out)
        grad_cpu = torch.sin(torch.arange(cpu.numel()).reshape_as(cpu).float())
        actual.backward(grad_cpu.to("musa"))
    stream.synchronize()
    expected = torch.softmax(cpu, -1)
    expected.backward(grad_cpu)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(x.grad.cpu(), cpu.grad, atol=1e-6, rtol=1e-5)
    assert returned.data_ptr() == out.data_ptr()


def test_softmax_unaligned_masked_rows_repeated():
    """A single invocation rarely exposes the muDNN 3.3.0 mp_31 failure."""
    columns = 537
    cpu = torch.sin(torch.arange(8128 * columns).reshape(8128, columns).float()) * 0.5
    cpu[:, 256:] = -10000.0
    expected = torch.softmax(cpu, -1).to("musa")
    x = cpu.to("musa")
    for _ in range(3000):
        actual = torch.softmax(x, -1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("functor", [torch.nn.Softmax, torch.nn.LogSoftmax])
def test_softmax_fp32(config, functor):
    shape, dim = config
    if functor == torch.nn.LogSoftmax:
        abs_diff = 1e-5
    else:
        abs_diff = 1e-6
    test = testing.OpTest(
        func=functor,
        input_args={"dim": dim},
        comparators=testing.DefaultComparator(abs_diff=abs_diff),
    )
    input_tensor = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    if input_tensor.dim() == 4 and random.random() < 0.5:
        input_tensor = input_tensor.to(memory_format=torch.channels_last)
    test.check_result({"input": input_tensor}, train=True)
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("functor", [torch.nn.Softmax, torch.nn.LogSoftmax])
def test_softmax_fp16(config, functor):
    shape, dim = config
    if functor == torch.nn.LogSoftmax:
        abs_diff, rel_diff = 5e-3, 5e-3
    else:
        abs_diff, rel_diff = 5e-4, 1e-3
    test = testing.OpTest(
        func=functor,
        input_args={"dim": dim},
        comparators=testing.DefaultComparator(abs_diff, rel_diff),
    )
    input_tensor = (
        torch.randn(shape, dtype=torch.float16).to(torch.float32).requires_grad_()
    )
    if input_tensor.dim() == 4 and random.random() < 0.5:
        input_tensor = input_tensor.to(memory_format=torch.channels_last)
    test.check_musafp16_vs_musafp32({"input": input_tensor}, train=True)
    test.check_grad_fn(fp16=True)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("functor", [torch.nn.Softmax, torch.nn.LogSoftmax])
def test_softmax_bf16(config, functor):
    shape, dim = config
    if functor == torch.nn.LogSoftmax:
        abs_diff, rel_diff = 1e-1, 1e-1
    else:
        abs_diff, rel_diff = 5e-2, 1e-5
    test = testing.OpTest(
        func=functor,
        input_args={"dim": dim},
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    input_tensor = torch.randn(shape, dtype=torch.bfloat16).requires_grad_()
    if input_tensor.dim() == 4 and random.random() < 0.5:
        input_tensor = input_tensor.to(memory_format=torch.channels_last)
    test.check_result({"input": input_tensor}, train=True)
    test.check_grad_fn(bf16=True)
