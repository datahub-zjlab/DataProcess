package org.datahub.service.imp;

import com.aliyun.oss.OSS;
import com.aliyun.oss.common.utils.HttpHeaders;
import com.aliyun.oss.model.OSSObject;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.datahub.utils.OSSFileUtil;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.InputStreamResource;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

@Service
@Slf4j
public class OSSDownloadService {
        @Getter
        @Setter
        private String BUCKET = "geocloud";
        @Getter
        private String pdfPrefix = "basic/ExampleForTest/PDF/";
        @Getter
        private String grobidPrefix = "basic/ExampleForTest/Grobid/";
        @Getter
        private String nougatPrefix = "basic/ExampleForTest/Nougat/";


        @Resource
        private OSS oss;

        public ByteArrayResource downloadOssFile(StringBuilder fileName,String fileType) throws IOException {
                try{
                        if(fileType.equalsIgnoreCase("Pdf")){
                                fileName.replace(0,fileName.length(),"01.pdf");
                                return download(pdfPrefix + fileName);
                        }
                        if(fileType.equalsIgnoreCase("Grobid")){
                                fileName.replace(0,fileName.length(), "01.mmd.final");
                                return download(grobidPrefix + fileName);
                        }
                        if(fileType.equalsIgnoreCase("Nougat")){
                                fileName.replace(0,fileName.length(), "01.mmd.final");
                                return download(nougatPrefix + fileName);
                        }

                }
                catch (Exception e){
                        log.error("");
                        throw e;
                }
                return null;
        }
        public ByteArrayResource download(String fileName) throws IOException {
                try{
                        OSSObject ossObject = readObject(fileName);
                        File temFile = OSSFileUtil.downloadOssFile(ossObject);
                        InputStream inputStream = Files.newInputStream(temFile.toPath());
                        byte[] body =new byte[inputStream.available()];
                        int num = inputStream.read(body);
//                        InputStreamResource = new org.springframework.core.io.InputStreamResource(inputStream);
                        inputStream.close();
                        return  new org.springframework.core.io.ByteArrayResource(body);
                }
                catch (Exception e){
                        log.error("");
                        throw e;
                }
        }


        public String getKey(String ossPath) {
                if (StringUtils.isBlank(ossPath)) {
                        return "";
                }
                ossPath = ossPath.replace("\\", "/");
                String prefix = String.format("oss://%s/", BUCKET);
                if (ossPath.startsWith(prefix)) {
                        return ossPath.replaceFirst(prefix, "");
                }
                return ossPath;
        }

        public OSSObject readObject(String path) {
                return oss.getObject(BUCKET, getKey(path));
        }
        public String readString(String ossPath) throws Exception {
                try (OSSObject ossObject = readObject(ossPath)) {
                        if (null == ossObject || null == ossObject.getObjectContent()) {
                                throw new Exception("invalid oss path");
                        } else {
                                try (InputStream content = ossObject.getObjectContent()) {
                                        return IOUtils.toString(content, StandardCharsets.UTF_8);
                                } catch (IOException e) {
                                        throw new Exception("读取oss文件io异常");
                                }
                        }
                }
        }

        public void writeObject(String path, InputStream is) throws Exception {
                oss.putObject(BUCKET, getKey(path), is);
        }

        public boolean isExist(String path) {
                return oss.doesObjectExist(BUCKET, path);
        }


}
