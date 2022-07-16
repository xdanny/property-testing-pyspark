import unittest

from pyspark.sql import SparkSession

class PySparkTestCase(unittest.TestCase):
    """Set-up of global test SparkSession"""

    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession
                     .builder
                     .master("local[1]")
                     .appName("PySpark unit test")
                     .getOrCreate())

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
