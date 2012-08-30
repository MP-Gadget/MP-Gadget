#ifndef F2C_INC
#define F2C_INC

#ifdef RANGER

#define COOLR coolr_
#define COOLI cooli_
#define COOLINMO coolinmo_
#define CHEMINMO cheminmo_
#define INIT_TOLERANCES init_tolerances_
#define EVOLVE_ABUNDANCES evolve_abundances_
#define RATE_EQ rate_eq_
#define CALC_GAMMA_TEMP calc_gamma_temp_
#define CALC_GAMMA calc_gamma_
#define CALC_SPEC calc_spec_
#define CALC_TEMP calc_temp_
#define LOAD_H2_TABLE load_h2_table_
#define OPEN_DEBUG_OUTPUT open_debug_output_
#define INIT_TEMPERATURE_LOOKUP init_temperature_lookup_
#define GET_ANGULAR_COORDS get_angular_coords_
#define GET_PIXELS_FOR_XYZ_AXES get_pixels_for_xyz_axes_
#define PROJECT_COLUMN project_column_
#define PROJECT project_
#define RAD_SOURCE_DATA rad_source_data_
#define THERMALINFO thermal_info_

#else

#ifdef VIP

#define COOLR coolr
#define COOLI cooli
#define COOLINMO coolinmo
#define CHEMINMO cheminmo
#define INIT_TOLERANCES init_tolerances
#define EVOLVE_ABUNDANCES evolve_abundances
#define RATE_EQ rate_eq
#define CALC_GAMMA_TEMP calc_gamma_temp
#define CALC_GAMMA calc_gamma
#define CALC_SPEC calc_spec
#define CALC_TEMP calc_temp
#define LOAD_H2_TABLE load_h2_table
#define OPEN_DEBUG_OUTPUT open_debug_output
#define INIT_TEMPERATURE_LOOKUP init_temperature_lookup
#define GET_ANGULAR_COORDS get_angular_coords
#define GET_PIXELS_FOR_XYZ_AXES get_pixels_for_xyz_axes
#define PROJECT_COLUMN project_column
#define PROJECT project
#define RAD_SOURCE_DATA rad_source_data
#define THERMALINFO thermal_info

#else

#define COOLR coolr_
#define COOLI cooli_
#define COOLINMO coolinmo_
#define CHEMINMO cheminmo_
#define INIT_TOLERANCES init_tolerances__
#define EVOLVE_ABUNDANCES evolve_abundances__
#define RATE_EQ rate_eq__
#define CALC_GAMMA_TEMP calc_gamma_temp__
#define CALC_GAMMA calc_gamma__
#define CALC_SPEC calc_spec__
#define CALC_TEMP calc_temp__
#define LOAD_H2_TABLE load_h2_table__
#define OPEN_DEBUG_OUTPUT open_debug_output__
#define INIT_TEMPERATURE_LOOKUP init_temperature_lookup__
#define GET_ANGULAR_COORDS get_angular_coords__
#define GET_PIXELS_FOR_XYZ_AXES get_pixels_for_xyz_axes__
#define PROJECT_COLUMN project_column__
#define PROJECT project_
#define RAD_SOURCE_DATA rad_source_data__
#define THERMALINFO thermal_info__

#endif

#endif

#endif /* F2C_INC */
