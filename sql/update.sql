UPDATE sf_activity_plan
   SET        
       START_DATE = cast(to_date('{sdate}','yyyy.mm.dd') as date),
       END_DATE = cast(to_date('{edate}','yyyy.mm.dd') as date),
       START_DATE_WORK = null,
       END_DATE_WORK = null,

       CONTRACTOR_ID = {execid},
       DISTRIBUTED={distributed}

 WHERE ACTIVITY_ID = {id}
   AND PLAN_ID = {plan}  