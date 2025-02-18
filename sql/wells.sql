  
select * from (SELECT exp.WELL_ID,
       exp.WELL_CODE,
       exp.COOP_ID,
       exp.ENTERPRISE_ID,
       exp.CITS_ID,
       exp.SHOP_ID,
       exp.CREW_ID,
       exp.FIELD_ID,
       exp.FIELD_CODE,
       coalesce(field_1485.identifier, exp.FIELD_NAME) FIELD_NAME,
       coalesce(field_1487.identifier, exp.FIELD_NAME) FIELD_NAME_EN,
       exp.POOL_ID,
       exp.WELL_PAD_ID,
       coalesce(wellpad_1485.identifier, EXP.WELL_PAD_NAME) WELL_PAD_NAME,
       coalesce(wellpad_1487.identifier, EXP.WELL_PAD_NAME) WELL_PAD_NAME_EN,
       coalesce(well_1485.identifier, exp.WELL_NAME) WELL_NAME,
       coalesce(well_1487.identifier, exp.WELL_NAME) WELL_NAME_EN,
       exp.STATUS_CODE,
       exp.PURPOSE_ID,
       exp.PURPOSE_CODE,
       exp.OPERATION_METHOD_CODE,
       exp.PODST_ID,
       exp.FEEDER_ID,
       exp.KTP_ID,
       exp.DNS_ID,
       exp.GZU_ID,
       exp.OTVOD_GZU_ID,
       exp.KNS_ID,
       exp.VRP_ID,
       exp.OTVOD_VRP_ID,
       exp.WP_REGIME_CURRENT,
       exp.OP_FOND,
       exp.DENSITY_OIL_REGIME_CURRENT,
       exp.BEHAVIOR_CODE,
       coord.y as LATITUDE,
       coord.x as LONGITUDE
  FROM "EXP#WELL_FOND" as exp
    LEFT JOIN "V#EARTH_SURF_FEATURE_ALIAS" as field_1485 ON field_1485.earth_surf_feature_s=exp.FIELD_ID AND field_1485.naming_sys_id=1485
    LEFT JOIN "V#EARTH_SURF_FEATURE_ALIAS" as field_1487 ON field_1487.earth_surf_feature_s=exp.FIELD_ID AND field_1487.naming_sys_id=1487
    LEFT JOIN "V#FACILITY_ALIAS" as wellpad_1485 ON wellpad_1485.aliased_object_s=exp.WELL_PAD_ID AND wellpad_1485.aliased_object_t='WELL_PAD' AND wellpad_1485.naming_sys_id=1485
    LEFT JOIN "V#FACILITY_ALIAS" as wellpad_1487 ON wellpad_1487.aliased_object_s=exp.WELL_PAD_ID AND wellpad_1487.aliased_object_t='WELL_PAD' AND wellpad_1487.naming_sys_id=1487
    LEFT JOIN "V#FACILITY_ALIAS" as well_1485 ON well_1485.aliased_object_s=exp.WELL_ID AND well_1485.aliased_object_t='WELL' AND well_1485.naming_sys_id=1485
    LEFT JOIN "V#FACILITY_ALIAS" as well_1487 ON well_1487.aliased_object_s=exp.WELL_ID AND well_1487.aliased_object_t='WELL' AND well_1487.naming_sys_id=1487
    LEFT JOIN p_well_coord coord on coord.well_id = exp.well_id and start_time <= now() and coalesce (end_time, cast('01.01.2100' as date)) > now() and coord.r_coord_sys_cnstr_id = 7664772222)
    x where x.field_id = {0} and  x.well_id in {1}
   
