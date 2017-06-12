package services;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.ResourceBundle;

import javax.imageio.ImageIO;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileUploadBase.SizeLimitExceededException;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.servlet.ServletFileUpload;
import org.json.JSONArray;
import org.json.JSONObject;

import com.google.cloud.vision.spi.v1.ImageAnnotatorClient;
import com.google.cloud.vision.v1.AnnotateImageRequest;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.cloud.vision.v1.BatchAnnotateImagesResponse;
import com.google.cloud.vision.v1.BoundingPoly;
import com.google.cloud.vision.v1.CropHint;
import com.google.cloud.vision.v1.CropHintsAnnotation;
import com.google.cloud.vision.v1.EntityAnnotation;
import com.google.cloud.vision.v1.Feature;
import com.google.cloud.vision.v1.Feature.Type;
import com.google.cloud.vision.v1.Image;
import com.google.protobuf.ByteString;

@SuppressWarnings("serial")
public class UploadServlet extends HttpServlet {

	private boolean isMultipart;
	private String uploadFilePath;
	private int maxFileSize = 3 * 1024 * 1024; // File should not be more than 3mb
	private int maxMemSize = 4 * 1024 * 1024; // File should not be more than 3mb
	private File file;
	private boolean isApproved;
	String approvedTags;
	String rejectedTags;
	String approvedString;

	public void init() {
		ResourceBundle configBundle = ResourceBundle.getBundle(Constants.CONFIG);
		uploadFilePath = configBundle.getString(Constants.UPLOAD_PATH);
		approvedTags = configBundle.getString(Constants.APPROVED_TAGS);
		rejectedTags = configBundle.getString(Constants.REJECTED_TAGS);
	}

	public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, java.io.IOException {
		
		// Initiate Response
		java.io.PrintWriter out = response.getWriter();
		response.setContentType("text/json");
		
		JSONObject resultData = new JSONObject();

		// Check whether the request is valid
		isMultipart = ServletFileUpload.isMultipartContent(request);
		if (!isMultipart) {
			resultData.put("message", "Not a valid request/file. Please try again.");
			out.println(resultData);
			return;
		}
		
		// Convert Tags to List

		HashMap<String,Float> approvedTagsList = new HashMap<String,Float>();
		String[] approvedTagArray = approvedTags.split("\\s*,\\s*");

		if(approvedTagArray.length > 0){
			for(int j=0;j<approvedTagArray.length;j++){
				String[] approvedTagParts = approvedTagArray[j].split("\\|");
				if(approvedTagParts.length == 2){
					approvedTagsList.put(approvedTagParts[0], Float.valueOf(approvedTagParts[1]));
				} else {
					resultData.put("message", "approved Tags are not configured correctly. Please use this example : approvedTags=person|90,man|90");
					out.println(resultData);
					return;
				}
			}
		} else {
			resultData.put("message", "Config file is not configured correctly");
			out.println(resultData);
			return;
		}
		
		List<String> rejectedTagsList = Arrays.asList(rejectedTags.split("\\s*,\\s*"));
		
		// Save the response to a file

		DiskFileItemFactory factory = new DiskFileItemFactory();
		factory.setSizeThreshold(maxMemSize);
		factory.setRepository(new File(uploadFilePath));

		ServletFileUpload upload = new ServletFileUpload(factory);
		upload.setSizeMax(maxFileSize);
		String fileName = "blob_" + System.currentTimeMillis() + ".png";

		try {
			List<FileItem> fileItems = upload.parseRequest(request);
			Iterator<FileItem> i = fileItems.iterator();
			
			while (i.hasNext()) {
				FileItem fi = (FileItem) i.next();

				if (!fi.isFormField()) {
					// Write the uploaded data request to file
					file = new File(uploadFilePath + fileName);
					fi.write(file);
				} else {
					// Obtain alternative tags and file request.
					if (fi.getFieldName().equals("approved") && !fi.getString().equals("")) {
						approvedTags = fi.getString();
					} else if (fi.getFieldName().equals("rejected") && !fi.getString().equals("")) {
						rejectedTags = fi.getString();
					} else if (fi.getFieldName().equals("file")) {
						// NOTE : This is not used currently
						file = new File(uploadFilePath + fileName);
						writeImageToDisk(fi, file);
					}
				}
			}
			
			// Start Google Vision API Recognition

			ImageAnnotatorClient vision = null;
			try {
				vision = ImageAnnotatorClient.create();
			} catch (IOException e1) {
				e1.printStackTrace(out);
				e1.printStackTrace();
			}

			// Obtain the image Data

			Path path = Paths.get(file.getAbsolutePath());
			byte[] data = null;
			try {
				data = Files.readAllBytes(path);
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			ByteString imgBytes = ByteString.copyFrom(data);

			// Adding all the features we require

			List<AnnotateImageRequest> requests = new ArrayList<>();
			Image img = Image.newBuilder().setContent(imgBytes).build();
			Feature labelFeature = Feature.newBuilder().setType(Type.LABEL_DETECTION).build();
			Feature textFeature = Feature.newBuilder().setType(Type.TEXT_DETECTION).build();
			Feature faceFeature = Feature.newBuilder().setType(Type.FACE_DETECTION).build();
			Feature landMarkFeature = Feature.newBuilder().setType(Type.LANDMARK_DETECTION).build();
			Feature safeSearchFeature = Feature.newBuilder().setType(Type.SAFE_SEARCH_DETECTION).build();
			Feature cropFeature = Feature.newBuilder().setType(Type.CROP_HINTS).build();
			
			// Prepare the request for recogntion by API 
			
			List<Feature> featuresList = new ArrayList<Feature>();
			featuresList.add(labelFeature);
			featuresList.add(textFeature);
			featuresList.add(faceFeature);
			featuresList.add(landMarkFeature);
			featuresList.add(safeSearchFeature);
			featuresList.add(cropFeature);

			AnnotateImageRequest annotationRequest = AnnotateImageRequest
														.newBuilder()
														.addAllFeatures(featuresList)
														.setImage(img)
														.build();
			requests.add(annotationRequest);

			// Execute and print the results

			BatchAnnotateImagesResponse batchImageResponse = vision.batchAnnotateImages(requests);
			List<AnnotateImageResponse> responses = batchImageResponse.getResponsesList();
			DecimalFormat f = new DecimalFormat("##.00");

			// Free memory
			imgBytes = null;
			
			// Analyze each response from API
			
			for (AnnotateImageResponse res : responses) {
				
				// Discard if the response has an error
				if (res.hasError()) {
					System.out.printf("Error: %s\n", res.getError().getMessage());
					return;
				}
				
				// Check if the response has any of the approved tags and doesn't have any rejected tags

				if(res.getLabelAnnotationsList().size() > 0){
					JSONArray labelArray = new JSONArray();
					
					for (EntityAnnotation annotation : res.getLabelAnnotationsList()) {
						if (approvedTagsList.containsKey(annotation.getDescription()) && ((annotation.getScore() * 100) >= approvedTagsList.get(annotation.getDescription()))) {
							isApproved = true;
							approvedString = annotation.getDescription();
						}
	
						if (rejectedTagsList.contains(annotation.getDescription())) {
							isApproved = false;
						}
						JSONObject label = new JSONObject();
						label.put("label", annotation.getDescription());
						label.put("score", f.format(annotation.getScore() * 100));
						
						labelArray.put(label);
					}
					
					resultData.put("labels", labelArray);
				}
				
				BoundingPoly faceBoundingPoly = null;
				
				resultData.put("faceCount", res.getFaceAnnotationsCount());
				if(res.getFaceAnnotationsCount() == 1){
					faceBoundingPoly = res.getFaceAnnotationsList().get(0).getBoundingPoly();
					resultData.put("faceVertices", res.getFaceAnnotationsList().get(0).toString());
				}

				// SafeSearch 0 - UNKNOWN, VERY_UNLIKELY - 1, UNLIKELY - 2, POSSIBLE - 3, LIKELY - 4, VERY_LIKELY - 5
				
				JSONObject safeSearch = new JSONObject();
				safeSearch.put("adult",res.getSafeSearchAnnotation().getAdultValue());
				safeSearch.put("spoof",res.getSafeSearchAnnotation().getSpoofValue());
				safeSearch.put("medical",res.getSafeSearchAnnotation().getMedicalValue());
				safeSearch.put("violence",res.getSafeSearchAnnotation().getViolenceValue());
				resultData.put("safeSearch", safeSearch);

				// An approved image must have only one face and should be safe
				if (res.getFaceAnnotationsCount() > 1 || res.getFaceAnnotationsCount() == 0 ||
						res.getSafeSearchAnnotation().getAdultValue() > 2 ||
						res.getSafeSearchAnnotation().getSpoofValue() > 2 ||
						res.getSafeSearchAnnotation().getMedicalValue() > 2 ||
						res.getSafeSearchAnnotation().getViolenceValue() > 2) {
					isApproved = false;
				}
				
				if(isApproved && faceBoundingPoly!=null){
					CropHintsAnnotation cropHintAnnotation = res.getCropHintsAnnotation();
					if(cropHintAnnotation!=null){
						byte[] imageData = null;
						try {
							imageData = Files.readAllBytes(path);
						} catch (IOException e) {
							e.printStackTrace();
						}
						
						List<CropHint> cropHints = cropHintAnnotation.getCropHintsList();
						System.out.println(cropHints);
						BufferedImage originalImage = createImageFromBytes(imageData);

//						System.out.println("-----------width : " + originalImage.getWidth() + ":" + originalImage.getHeight());
//						System.out.println(cropHints.get(0).getBoundingPoly().getVertices(0).getX() + "---" +
//												cropHints.get(0).getBoundingPoly().getVertices(0).getY() + "---" +
//												(cropHints.get(0).getBoundingPoly().getVertices(2).getX()) + "---" +
//												(cropHints.get(0).getBoundingPoly().getVertices(2).getY()));
//						
//						BufferedImage croppedImage = originalImage.getSubimage(cropHints.get(0).getBoundingPoly().getVertices(0).getX(),
//												cropHints.get(0).getBoundingPoly().getVertices(0).getY(),
//												cropHints.get(0).getBoundingPoly().getVertices(2).getX() - cropHints.get(0).getBoundingPoly().getVertices(0).getX(),
//												cropHints.get(0).getBoundingPoly().getVertices(2).getY() - cropHints.get(0).getBoundingPoly().getVertices(0).getY());
//						
						int margin = Constants.IMAGE_MARGIN;
						int startX = ((faceBoundingPoly.getVertices(0).getX()-margin) > 0) ? (faceBoundingPoly.getVertices(0).getX()-margin) : faceBoundingPoly.getVertices(0).getX(); 
						int startY = ((faceBoundingPoly.getVertices(0).getY()-margin) > 0) ? (faceBoundingPoly.getVertices(0).getY()-margin) : faceBoundingPoly.getVertices(0).getY();
						int width = faceBoundingPoly.getVertices(2).getX() - faceBoundingPoly.getVertices(0).getX();
						int height = faceBoundingPoly.getVertices(2).getY() - faceBoundingPoly.getVertices(0).getY();
						int endX = ((width + margin) < originalImage.getWidth()) ?(width + margin) : width;
						int endY = ((height + margin) < originalImage.getHeight()) ?(height + margin) : height;
						
						
						BufferedImage croppedImage = originalImage.getSubimage(startX,startY,endX,endY);
						
						File outputfile = new File(uploadFilePath + "cropped_" + fileName);
						ImageIO.write(croppedImage, "png", outputfile);
					}
				}
			}

			// Prepare Final Response and output the response
			resultData.put("isApproved", isApproved);
			
			if(isApproved){
				resultData.put("approvedTag", approvedString);
			}
			
			out.println(resultData);
			
			// Delete the file if it's not approved
			if(!isApproved){
				file.delete();
			}

		} catch(SizeLimitExceededException ex){
			out.println("<p>Image is larger than the max fileUpload limit : "+ ex + "</p>");
			ex.printStackTrace(out);
			if(file!=null){
				file.delete();
			}
		} catch (Exception ex) {
			ex.printStackTrace(out);
			ex.printStackTrace();
			if(file!=null){
				file.delete();
			}
		}
	}

	public void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, java.io.IOException {
		throw new ServletException("GET method used with " + getClass().getName() + ": POST method required.");
	}

	/*
	 * writeImageToDisk - Writes the fileItem to a file 
	 * 		Currently this is not used
	 */
	public boolean writeImageToDisk(FileItem item, File imageFile) {
		String errorMessage = null;
		FileOutputStream out = null;
		boolean ret = false;
		try {
			out = createOutputStream(imageFile);

			byte[] headerBytes = new byte[22];
			InputStream imageStream = item.getInputStream();
			imageStream.read(headerBytes);

			String header = new String(headerBytes);

			byte[] b = new byte[4 * 1024];
			byte[] decoded;
			int read = 0;
			while ((read = imageStream.read(b)) != -1) {
				if (Base64.isArrayByteBase64(b)) {
					decoded = Base64.decodeBase64(b);
					out.write(decoded);
				}
			}
			ret = true;
		} catch (IOException e) {
			StringWriter sw = new StringWriter();
			e.printStackTrace(new PrintWriter(sw));
			errorMessage = "error: " + sw;
		} finally {
			if (out != null) {
				try {
					out.close();
				} catch (Exception e) {
					StringWriter sw = new StringWriter();
					e.printStackTrace(new PrintWriter(sw));
					System.out.println("Cannot close outputStream after writing file to disk!" + sw.toString());
				}
			}

		}

		return ret;
	}

	protected FileOutputStream createOutputStream(File imageFile) throws FileNotFoundException {
		imageFile.getParentFile().mkdirs();
		return new FileOutputStream(imageFile);
	}
	
	private BufferedImage createImageFromBytes(byte[] imageData) {
	    ByteArrayInputStream bais = new ByteArrayInputStream(imageData);
	    try {
	        return ImageIO.read(bais);
	    } catch (IOException e) {
	        throw new RuntimeException(e);
	    }
	}
}