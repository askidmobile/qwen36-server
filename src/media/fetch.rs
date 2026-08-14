use super::{MediaError, MediaErrorKind};
use futures::StreamExt;
use reqwest::header::{CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, LOCATION};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::{Duration, Instant};
use tokio::net::lookup_host;
use url::Url;

const MAX_REDIRECTS: usize = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const TOTAL_TIMEOUT: Duration = Duration::from_secs(30);

pub async fn fetch_public_https(
    url: &str,
    max_bytes: u64,
) -> Result<(Vec<u8>, String), MediaError> {
    fetch_with_deadline(url, max_bytes, Instant::now() + TOTAL_TIMEOUT).await
}

async fn fetch_with_deadline(
    initial: &str,
    max_bytes: u64,
    deadline: Instant,
) -> Result<(Vec<u8>, String), MediaError> {
    let mut current = Url::parse(initial).map_err(|_| MediaError::invalid("invalid media URL"))?;
    for redirects in 0..=MAX_REDIRECTS {
        validate_url(&current)?;
        let host = current
            .host_str()
            .ok_or_else(|| MediaError::invalid("media URL has no host"))?;
        let port = current.port_or_known_default().unwrap_or(443);
        let addresses = lookup_host((host, port))
            .await
            .map_err(|_| MediaError::invalid("media URL DNS resolution failed"))?
            .collect::<Vec<_>>();
        if addresses.is_empty() || addresses.iter().any(|address| !is_public_ip(address.ip())) {
            return Err(MediaError::invalid("media URL destination is not public"));
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(remaining(deadline)?)
            .resolve_to_addrs(host, &addresses)
            .build()
            .map_err(|_| MediaError::new(MediaErrorKind::Internal, "cannot build media client"))?;
        let response = client
            .get(current.clone())
            .header(reqwest::header::ACCEPT_ENCODING, "identity")
            .send()
            .await
            .map_err(map_reqwest_error)?;
        if response.status().is_redirection() {
            if redirects == MAX_REDIRECTS {
                return Err(MediaError::invalid("too many media URL redirects"));
            }
            let location = response
                .headers()
                .get(LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| MediaError::invalid("media redirect has no valid Location"))?;
            current = current
                .join(location)
                .map_err(|_| MediaError::invalid("invalid media redirect URL"))?;
            continue;
        }
        if !response.status().is_success() {
            return Err(MediaError::invalid("media URL returned non-success status"));
        }
        if response
            .headers()
            .get(CONTENT_ENCODING)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
        {
            return Err(MediaError::invalid(
                "compressed media HTTP response is forbidden",
            ));
        }
        if response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .is_some_and(|length| length > max_bytes)
        {
            return Err(MediaError::new(
                MediaErrorKind::EncodedTooLarge,
                "media URL exceeds encoded size limit",
            ));
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .unwrap_or("application/octet-stream")
            .trim()
            .to_ascii_lowercase();
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = tokio::time::timeout(remaining(deadline)?, stream.next())
            .await
            .map_err(|_| MediaError::new(MediaErrorKind::Timeout, "media download timed out"))?
        {
            let chunk = chunk.map_err(map_reqwest_error)?;
            let next = bytes.len().checked_add(chunk.len()).ok_or_else(|| {
                MediaError::new(MediaErrorKind::EncodedTooLarge, "media too large")
            })?;
            if u64::try_from(next).unwrap_or(u64::MAX) > max_bytes {
                return Err(MediaError::new(
                    MediaErrorKind::EncodedTooLarge,
                    "media URL exceeds encoded size limit",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        return Ok((bytes, content_type));
    }
    Err(MediaError::invalid("media redirect failure"))
}

fn remaining(deadline: Instant) -> Result<Duration, MediaError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| MediaError::new(MediaErrorKind::Timeout, "media download timed out"))
}

fn validate_url(url: &Url) -> Result<(), MediaError> {
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(MediaError::invalid(
            "media URL must be credential-free HTTPS",
        ));
    }
    Ok(())
}

fn map_reqwest_error(error: reqwest::Error) -> MediaError {
    if error.is_timeout() {
        MediaError::new(MediaErrorKind::Timeout, "media download timed out")
    } else {
        MediaError::invalid("media download failed")
    }
}

pub fn is_public_ip(ip: IpAddr) -> bool {
    match normalize_ip(ip) {
        IpAddr::V4(ip) => is_public_v4(ip),
        IpAddr::V6(ip) => is_public_v6(ip),
    }
}

fn normalize_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .map(IpAddr::V4)
            .unwrap_or(IpAddr::V6(ip)),
        other => other,
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    !ip.is_unspecified()
        && !ip.is_loopback()
        && !ip.is_private()
        && !ip.is_link_local()
        && !ip.is_multicast()
        && !ip.is_broadcast()
        && !(octets[0] == 0)
        && !(octets[0] == 100 && (64..=127).contains(&octets[1]))
        && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        && !(octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        && !(octets[0] == 192 && octets[1] == 88 && octets[2] == 99)
        && !(octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
        && !(octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        && !(octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
        && octets[0] < 240
}

fn is_public_v6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    !ip.is_unspecified()
        && !ip.is_loopback()
        && !ip.is_multicast()
        && !(segments[0] & 0xfe00 == 0xfc00)
        && !(segments[0] & 0xffc0 == 0xfe80)
        && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
        && !(segments[0] == 0x2001 && segments[1] == 0x0002)
        && !(segments[0] == 0x2001 && segments[1] == 0x0010)
        && !(segments[0] == 0x2001 && segments[1] == 0x0020)
        && !(segments[0] == 0x2001 && segments[1] == 0x0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_and_reserved_addresses_are_rejected() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "100.64.0.1",
            "169.254.169.254",
            "192.0.2.1",
            "198.18.0.1",
            "203.0.113.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "2001:db8::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(!is_public_ip(ip.parse().unwrap()), "{ip}");
        }
        assert!(is_public_ip("1.1.1.1".parse().unwrap()));
        assert!(is_public_ip("2606:4700:4700::1111".parse().unwrap()));
    }
}
