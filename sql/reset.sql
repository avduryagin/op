UPDATE sf_activity_plan
   SET        
       START_DATE = to_date('01.01.1970', 'dd.mm.yyyy'),
       END_DATE = to_date('01.01.1970', 'dd.mm.yyyy'),
       CONTRACTOR_ID = {execid},
       DISTRIBUTED={distributed}

 WHERE ACTIVITY_ID = {id}
   AND PLAN_ID = {plan}  