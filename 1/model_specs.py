from pygam import LinearGAM, s, te, l

__all__ = ["build_model_specs"]

def build_model_specs(feature_names, n_splines_s=20, n_splines_te=10):
    core3 = feature_names[:3]
    has_preg = len(feature_names) > 3
    has_birth = len(feature_names) > 4

    specs = {
        'M1_baseline': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s)
            ),
            'features': core3
        },
        'M2_interact_WB': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s)
                + te(0, 1, n_splines=n_splines_te)
            ),
            'features': core3
        },
        'M3_interact_WA': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s)
                + te(0, 2, n_splines=n_splines_te)
            ),
            'features': core3
        },
        'M4_interact_BA': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s)
                + te(1, 2, n_splines=n_splines_te)
            ),
            'features': core3
        },
        'M5_all_pairwise_interactions': {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s)
                + te(0, 1, n_splines=n_splines_te) + te(0, 2, n_splines=n_splines_te) + te(1, 2, n_splines=n_splines_te)
            ),
            'features': core3
        }
    }

    if has_preg:
        specs['M6_add_pregnancy_linear'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3)
            ),
            'features': core3 + [feature_names[3]]
        }
    if has_birth:
        specs['M7_add_birth_linear'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3)
            ),
            'features': core3 + [feature_names[4]]
        }
    if has_preg and has_birth:
        specs['M8_add_preg_birth_linear'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3) + l(4)
            ),
            'features': core3 + [feature_names[3], feature_names[4]]
        }
        specs['M9_full_main_linear_counts_all_interactions'] = {
            'builder': lambda: LinearGAM(
                s(0, n_splines=n_splines_s) + s(1, n_splines=n_splines_s) + s(2, n_splines=n_splines_s) + l(3) + l(4)
                + te(0, 1, n_splines=n_splines_te) + te(0, 2, n_splines=n_splines_te) + te(1, 2, n_splines=n_splines_te)
            ),
            'features': core3 + [feature_names[3], feature_names[4]]
        }
    return specs
