from evaluation.results import packageResults


print(packageResults.modelLossResults.getDictionary())

x = {
    'mnistTransformedLimitedRotation': {
        'conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce': {
            'train': {'kl': 14.010067552947998, 'reconstruction': 200.85263961181641, 'total': 214.8627073059082},
            'test': {'kl': 13.903411302566528, 'reconstruction': 200.44749969482422, 'total': 214.35091104125976},
            'val': {'reconstruction': 200.66045297241212, 'kl': 14.107191741943359, 'total': 214.76764404296875}
        }, 'deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce': {
            'train': {'reconstruction': 112.7609843383789, 'kl': 30.183209438323974, 'total': 142.9441939025879},
            'val': {'kl': 29.953226615905763, 'reconstruction': 115.72759242248536, 'total': 145.6808188171387},
            'test': {'reconstruction': 114.6999822845459, 'kl': 30.715065273284914, 'total': 145.4150477142334}
        }
    }, 'mnist': {
        'conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce': {
            'train': {'reconstruction': 70.3729598236084, 'kl': 24.024353672027587, 'total': 94.3973134918213},
            'val': {'kl': 23.904631633758544, 'reconstruction': 72.69932510375976, 'total': 96.6039567565918},
            'test': {'reconstruction': 72.13639793395996, 'kl': 23.895653228759766, 'total': 96.03205146789551}
        }, 'deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce': {
            'train': {'kl': 22.65600029373169, 'reconstruction': 68.40176537322998, 'total': 91.0577657623291},
            'test': {'kl': 22.52736291885376, 'reconstruction': 72.98750297546387, 'total': 95.51486564636231},
            'val': {'reconstruction': 73.50605361938477, 'kl': 22.493740787506102, 'total': 95.9997947692871}
        }
    }, 'norb': {
        'conv_96x96_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce': {
            'train': {'kl': 12.219338516008847, 'reconstruction': 4437.136887825635, 'total': 4449.356222342211},
            'test': {'kl': 12.023562600092633, 'reconstruction': 4481.825707204057, 'total': 4493.849272300186},
            'val': {'reconstruction': 4457.500487778903, 'kl': 12.051928769413827, 'total': 4469.552407095953}
        },
        'deepDense_96x96_ENC_1024_2048_2048_DEC_2048_2048_1024_LAT_32_bce': {
            'train': {'reconstruction': 4796.150700142494, 'kl': 20.095398935971716, 'total': 4816.24610437105},
            'val': {'kl': 20.109953224904253, 'reconstruction': 4804.706780327691, 'total': 4824.816711425781},
            'test': {'reconstruction': 4828.895692676183, 'kl': 20.185437963823233, 'total': 4849.081124011381}
        }
    }, 'shapesTransformedLimitedRotation': {
        'conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce': {
             'train': {'kl': 19.02814576983452, 'reconstruction': 180.22555416107178, 'total': 199.25370008945464},
             'test': {'kl': 18.93033010363579, 'reconstruction': 180.38353855609893, 'total': 199.31386737823487},
             'val': {'reconstruction': 180.4216493844986, 'kl': 18.84674108028412, 'total': 199.26838972568513}
        }, 'deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce': {
            'train': {'reconstruction': 62.4551468038559, 'kl': 19.855650840997697, 'total': 82.31079781532287},
            'val': {'kl': 19.80926078557968, 'reconstruction': 62.55390862226486, 'total': 82.36316933631898},
            'test': {'reconstruction': 62.459310591220856, 'kl': 19.82650119662285, 'total': 82.28581192493439}
        }
    }
}
