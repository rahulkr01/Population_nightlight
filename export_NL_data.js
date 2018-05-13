var imageCollection = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS"),
    table3 = ee.FeatureCollection("users/rahulkraj/Census_2011"),
    image = ee.Image("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS/F182013");

Map.addLayer(image,{bands:['stable_lights']});
Map.addLayer(table3,{color:'green'});
var sumOfLights = image.reduceRegions({
  collection:table2,
  reducer:ee.Reducer.mean(),
});
print(sumOfLights);

Export.table.toDrive({
  collection: sumOfLights,
  description:'SumofLights',
  fileFormat: 'CSV'
});



