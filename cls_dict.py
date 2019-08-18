import json

num_label_dict = {
    0: 'background', 1: 'brake_warning', 2: 'abs_light', 3: 'brake_pad_warning_1', 4: 'press_clutch_pedal_light_1',
    5: 'hill_descent_control_2', 6: 'press_clutch_pedal_light', 7: 'parking_brake_light', 8: 'parking_brake_light_1',
    9: 'low_fuel_warning', 10: 'torching_light', 11: 'code_pressure_low_warning', 12: 'oil_pressure_low',
    13: 'oil_pressure_low_1', 14: 'not_buckle_up', 15: 'engine_emissions_warning', 16: 'parking_auxiliary_indicator',
    17: 'airbag_deactived', 18: 'power_steering_warning', 19: 'steering_lock_warning_1', 20: 'steering_lock_warning',
    21: 'speed_limiter_light', 22: 'auto_pilot_light', 23: 'auto_pilot_light_off', 24: 'brake_hold_light_1',
    25: 'automatic_shutoff_indicator', 26: 'information_indicator', 27: 'triangle_warning_light',
    28: 'service_required_1', 29: 'auto_gear_warning', 30: 'tire_pressure_low', 31: 'torching_control_off_light_1',
    32: 'stabillity_light', 33: 'stabillity_off_light', 34: 'temperature_warning', 35: 'low_temperature_indicator1',
    36: 'gear_temperature_warning', 37: 'door_open_light', 38: 'lane_departure_warning_1',
    39: 'lane_maintenance_Indicator', 40: 'crach_warning_light', 41: 'parking_assist', 42: 'sandy__indicator',
    43: 'battery_warning', 44: 'battery_temp_warning', 45: 'auto_head_light', 46: 'sidelight_info',
    47: 'fog_light_front', 48: 'fog_light_back', 49: 'day_light_indicator', 50: 'low_light_indicator',
    51: 'high_beam_headlights', 52: 'auto_head_light_warning_1', 53: 'auto_head_light_1', 54: 'auto_head_light_warning',
    55: 'distance_warning', 56: 'brake_hold_light_2', 57: 'brake_hold_light', 58: 'washer_fluid_low_1',
    59: 'washer_fluid_low_2', 60: 'direction_light_left', 61: 'direction_light_right', 62: 'night_mode_light',
    63: 'lane_departure_warning', 64: 'snow_mode_light_2', 65: 'snow_mode_light_1', 66: 'suspension_warning',
    67: 'fuel_filter_warning', 68: 'dirver_tired_warning', 69: 'bolb_warning', 70: 'bolb_warning_3',
    71: 'hill_descent_control', 72: 'engine_limit_lamp', 73: 'ev_mode_light', 74: 'ev_mode_light_1',
    75: 'trunk_open_light', 76: 'bonnet_open_light', 77: 'smart_key_warning_1', 78: 'key_fob_battery_warning',
    79: 'key_not_in_vehicle', 80: 'engine_safe_light_2', 81: 'safe_light', 82: 'engine_pre_heat_light',
    83: 'rear_window_defrost', 84: 'catalytic_converter_warning', 85: 'eco_light_2', 86: 'service_required',
    87: '4wd_mode_light', 88: '4wd_lock_light_1', 89: '4steering_light', 90: '4wd_mode_warning_1',
    91: '4wd_mode_warning', 92: '4lo_light', 93: '4lo_light_1', 94: 'ready_light', 95: 'epb_light',
    96: 'snow_mode_light', 97: 'ebd_light', 98: 'limit_speed_light', 99: 'at_oil_temp_warning',
    100: 'power_steering_warning_1', 101: 'vdc_off_light_1', 102: 'esp_light', 103: 'eps_light', 104: 'epc_light',
    105: 'svs_warning', 106: 'eco_light_3', 107: 'eco_light_1', 108: 'cvt_check', 109: 'sport_mode_light',
    110: 'abc_light', 111: 'airbag_light_1', 112: 'speed_limiter_warning', 113: 'at_warning', 114: 'afs_off_light',
    115: 'airbag_light', 116: 'cruise_control_light', 117: 'cruise_main_light', 118: 'smart_key_warning',
    119: 'acc_control_light', 120: 'dct_light', 121: 'torching_control_off_light', 122: 'tire_pressure_check',
    123: 'torching_control_light', 124: 'i_stop', 125: 'trc_off', 126: 'vsc_off_light', 127: 'vsc_on_light', 128: 'bsm'
}

label_num_dict = {
    'background': 0, 'brake_warning': 1, 'abs_light': 2, 'brake_pad_warning_1': 3, 'press_clutch_pedal_light_1': 4,
    'hill_descent_control_2': 5, 'press_clutch_pedal_light': 6, 'parking_brake_light': 7, 'parking_brake_light_1': 8,
    'low_fuel_warning': 9, 'torching_light': 10, 'code_pressure_low_warning': 11, 'oil_pressure_low': 12,
    'oil_pressure_low_1': 13, 'not_buckle_up': 14, 'engine_emissions_warning': 15, 'parking_auxiliary_indicator': 16,
    'airbag_deactived': 17, 'power_steering_warning': 18, 'steering_lock_warning_1': 19, 'steering_lock_warning': 20,
    'speed_limiter_light': 21, 'auto_pilot_light': 22, 'auto_pilot_light_off': 23, 'brake_hold_light_1': 24,
    'automatic_shutoff_indicator': 25, 'information_indicator': 26, 'triangle_warning_light': 27,
    'service_required_1': 28, 'auto_gear_warning': 29, 'tire_pressure_low': 30, 'torching_control_off_light_1': 31,
    'stabillity_light': 32, 'stabillity_off_light': 33, 'temperature_warning': 34, 'low_temperature_indicator1': 35,
    'gear_temperature_warning': 36, 'door_open_light': 37, 'lane_departure_warning_1': 38,
    'lane_maintenance_Indicator': 39, 'crach_warning_light': 40, 'parking_assist': 41, 'sandy__indicator': 42,
    'battery_warning': 43, 'battery_temp_warning': 44, 'auto_head_light': 45, 'sidelight_info': 46,
    'fog_light_front': 47, 'fog_light_back': 48, 'day_light_indicator': 49, 'low_light_indicator': 50,
    'high_beam_headlights': 51, 'auto_head_light_warning_1': 52, 'auto_head_light_1': 53, 'auto_head_light_warning': 54,
    'distance_warning': 55, 'brake_hold_light_2': 56, 'brake_hold_light': 57, 'washer_fluid_low_1': 58,
    'washer_fluid_low_2': 59, 'direction_light_left': 60, 'direction_light_right': 61, 'night_mode_light': 62,
    'lane_departure_warning': 63, 'snow_mode_light_2': 64, 'snow_mode_light_1': 65, 'suspension_warning': 66,
    'fuel_filter_warning': 67, 'dirver_tired_warning': 68, 'bolb_warning': 69, 'bolb_warning_3': 70,
    'hill_descent_control': 71, 'engine_limit_lamp': 72, 'ev_mode_light': 73, 'ev_mode_light_1': 74,
    'trunk_open_light': 75, 'bonnet_open_light': 76, 'smart_key_warning_1': 77, 'key_fob_battery_warning': 78,
    'key_not_in_vehicle': 79, 'engine_safe_light_2': 80, 'safe_light': 81, 'engine_pre_heat_light': 82,
    'rear_window_defrost': 83, 'catalytic_converter_warning': 84, 'eco_light_2': 85, 'service_required': 86,
    '4wd_mode_light': 87, '4wd_lock_light_1': 88, '4steering_light': 89, '4wd_mode_warning_1': 90,
    '4wd_mode_warning': 91, '4lo_light': 92, '4lo_light_1': 93, 'ready_light': 94, 'epb_light': 95,
    'snow_mode_light': 96, 'ebd_light': 97, 'limit_speed_light': 98, 'at_oil_temp_warning': 99,
    'power_steering_warning_1': 100, 'vdc_off_light_1': 101, 'esp_light': 102, 'eps_light': 103, 'epc_light': 104,
    'svs_warning': 105, 'eco_light_3': 106, 'eco_light_1': 107, 'cvt_check': 108, 'sport_mode_light': 109,
    'abc_light': 110, 'airbag_light_1': 111, 'speed_limiter_warning': 112, 'at_warning': 113, 'afs_off_light': 114,
    'airbag_light': 115, 'cruise_control_light': 116, 'cruise_main_light': 117, 'smart_key_warning': 118,
    'acc_control_light': 119, 'dct_light': 120, 'torching_control_off_light': 121, 'tire_pressure_check': 122,
    'torching_control_light': 123, 'i_stop': 124, 'trc_off': 125, 'vsc_off_light': 126, 'vsc_on_light': 127, 'bsm': 128
}



if __name__ == '__main__':

    label_num_dict = {}
    for key, value in num_label_dict.items():
        label_num_dict[value] = key

    print(label_num_dict)
