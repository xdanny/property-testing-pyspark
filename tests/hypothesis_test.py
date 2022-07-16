import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


from .pyspark_test_case import PySparkTestCase

from datetime import datetime, timedelta
import hypothesis.strategies as st
from hypothesis import given, settings, HealthCheck
from typing import List, NamedTuple
from tinsel import struct, transform
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


settings.register_profile(
    "my_profile",
    max_examples=50,
    deadline=60 * 1000,  # Allow 1 min per example (deadline is specified in milliseconds)
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)

@struct
class OrderItem(NamedTuple):
  item_id: str
  item_value: float

@struct
class Order(NamedTuple):
  order_id: str
  created_at: datetime
  updated_at: datetime
  customer_id: str
  order_items: List[OrderItem]

# COMMAND ----------

week_sunday = (datetime.now() - timedelta(days = datetime.now().weekday() + 1))
time_window = (datetime.now() - timedelta(days = datetime.now().weekday() + 7))

def order_strategy():
    return st.lists(st.builds(Order,
                     order_id=st.text(min_size=5),
                     created_at=st.datetimes(min_value=time_window, max_value=week_sunday),
                     updated_at=st.datetimes(min_value=time_window, max_value=week_sunday),
                     customer_id=st.text(min_size=5),
                     order_items=st.lists(st.builds(OrderItem,
                                           item_id=st.text(min_size=5),
                                           item_value=st.floats(min_value=0, max_value=100)), min_size=1)), min_size=5)


schema = transform(Order)

def test_dataframe_transformations(df: DataFrame) -> DataFrame:
    temp_df = df.select('*', F.explode_outer('order_items'))\
        .select(F.col('created_at'), F.col('col.item_value').alias('item_value'))\
        .withColumn('day', F.to_date('created_at'))\
        .withColumn('week', F.weekofyear('created_at'))

    weekly_revenue_df = temp_df.groupBy('week').agg(
        F.sum('item_value').alias('weekly_revenue')
    )

    return temp_df.groupBy('week', 'day').agg(
        F.sum('item_value').alias('daily_revenue')
    ).join(weekly_revenue_df, on='week', how='inner')


class SimpleTestCase(PySparkTestCase):

    def test_spark(self):
        @given(data=order_strategy())
        @settings(settings.load_profile("my_profile"))
        def test_samples(data):
            df = self.spark.createDataFrame(data=data, schema=schema, verifySchema=False)
            collected_list = test_dataframe_transformations(df).collect()

            for row in collected_list:
                assert row['weekly_revenue'] >= row['daily_revenue']


        test_samples()
