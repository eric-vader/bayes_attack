import torch.fft
torch.manual_seed(12345)
spectrum = torch.randn(1, 3, 10, 10, 6, dtype=torch.cfloat)
spectrum[..., 0].imag.fill_(0)
spectrum[..., -1].imag.fill_(0)
print(spectrum.size())
spectrum = torch.view_as_real(torch.fft.fftn(spectrum, dim=(2, 3)))
print(spectrum.size())
recovered_old = torch.irfft(spectrum, 3, normalized=True, signal_sizes=(10, 10, 10))

recovered_new = torch.fft.irfftn(torch.complex(spectrum[..., 0], spectrum[..., 1]), dim=(2, 3, 4), s=(10, 10, 10), norm='ortho')
print((recovered_old - recovered_new).abs().max())