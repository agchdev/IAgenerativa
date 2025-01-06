import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    El generador toma como entrada un vector de ruido (z) y produce una imagen.
    Para DCGAN, se utilizan capas ConvTranspose2d para "ampliar" ese vector
    en dimensiones de anchura y altura.
    """

    def __init__(self, nz=100, ngf=64, nc=3):
        """
        Parámetros:
        ----------
        - nz: Tamaño del vector de ruido (típicamente 100).
        - ngf: Número base de filtros en el generador (64 es común en DCGAN).
        - nc: Número de canales de salida (3 para RGB, 1 para blanco y negro).
        """
        super(Generator, self).__init__()
        
        # Bloque 1: (nz) -> (ngf*8, 4, 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),  # ReLU con in-place

            # Bloque 2: (ngf*8, 4, 4) -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Bloque 3: (ngf*4, 8, 8) -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Bloque 4: (ngf*2, 16, 16) -> (ngf, 32, 32)
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Bloque 5: (ngf, 32, 32) -> (nc, 64, 64)
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  
            # Tanh para que los valores de cada píxel estén en el rango [-1, 1].
        )

    def forward(self, input):
        """
        input: tensor de forma (batch_size, nz, 1, 1)
        output: tensor (batch_size, nc, 64, 64)
        """
        return self.main(input)
